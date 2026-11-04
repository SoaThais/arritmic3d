#ifndef FIM_EIKONAL_PROPAGATION_IMPL_H
#define FIM_EIKONAL_PROPAGATION_IMPL_H

#include <cmath>
#include <limits>
#include <cstdlib>

namespace fim_eikonal_detail {
    constexpr double INF = std::numeric_limits<double>::infinity();
}

// ---------------------------------------------------------------------------
// Inicialização e suporte
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::Initialize(Tissue& tissue) {

    for (const Node& node : tissue.tissue_nodes) {
        if (node.type != CELL_TYPE_VOID && node.parameters != nullptr && node.parameters->min_potential > 0.0f)
            throw std::runtime_error("FIMEikonalPropagation: min_potential > 0 is not supported by this solver.");
    }

    states_.assign(tissue.tissue_nodes.size(), PropagationState{});
    list_.clear();
    pending_.clear();
    seeds_.clear();
    recovery_heap_ = RecoveryHeap();

    BuildStencil(tissue);
}

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::CheckSupported(const Tissue& tissue) const {

    if (tissue.long_apd_reactivation)
        throw std::runtime_error("FIMEikonalPropagation: long_apd_reactivation is not supported by this solver.");
}

/**
 * Posição dos vizinhos e segmentos (arestas) do estêncil.
 *
 * Os vizinhos de Geometry são gerados com os laços z, y, x (de -D a D, sem o
 * centro). Reproduzimos essa ordem para obter os deslocamentos inteiros e
 * conferimos com geometry.displacement.
 */
template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::BuildStencil(Tissue& tissue) {

    const auto& g = tissue.tissue_geometry;
    constexpr int    D = Geometry::distance;
    constexpr size_t K = Geometry::num_neighbours;

    vector<std::array<int, 3>> offsets;
    offsets.reserve(K);
    stencil_pos_.clear();
    stencil_pos_.reserve(K);

    size_t pos = 0;
    for (int i = -D; i <= D; ++i) {          // z
        for (int j = -D; j <= D; ++j) {      // y
            for (int k = -D; k <= D; ++k) {  // x

                if (i == 0 && j == 0 && k == 0)
                    continue;

                const int expected = i * g.ext_size_x * g.ext_size_y + j * g.ext_size_x + k;

                if (g.displacement[pos] != expected)
                    throw std::runtime_error("FIMEikonalPropagation::BuildStencil: unexpected neighbour ordering in Geometry.");

                offsets.push_back({k, j, i});
                stencil_pos_.push_back(g.relative_position[pos].template cast<double>());
                ++pos;
            }
        }
    }

    auto shell = [](const std::array<int, 3>& o) {
        return std::max({std::abs(o[0]), std::abs(o[1]), std::abs(o[2])});
    };

    edges_.clear();

    for (size_t a = 0; a < K; ++a) {
        for (size_t b = a + 1; b < K; ++b) {

            const int step = std::abs(offsets[a][0] - offsets[b][0])
                           + std::abs(offsets[a][1] - offsets[b][1])
                           + std::abs(offsets[a][2] - offsets[b][2]);

            if (step == 1 && shell(offsets[a]) == shell(offsets[b]))
                edges_.emplace_back(a, b);
        }
    }
}

template <typename APM, typename CVM>
std::size_t FIMEikonalPropagation<APM, CVM>::NodeIndex(const Tissue& tissue, const Node* node) const {
    assert(node >= tissue.tissue_nodes.data());
    assert(node < tissue.tissue_nodes.data() + tissue.tissue_nodes.size());
    return static_cast<std::size_t>(node - tissue.tissue_nodes.data());
}

template <typename APM, typename CVM>
float FIMEikonalPropagation<APM, CVM>::SafetyFactor(const Node& node) {
    return (node.parameters != nullptr) ? node.parameters->safety_factor : 1.0f;
}

template <typename APM, typename CVM>
bool FIMEikonalPropagation<APM, CVM>::HasPendingWork() const {

    if (!seeds_.empty() || !recovery_heap_.empty())
        return true;

    for (std::size_t j : list_)
        if (states_[j].status == Status::LIST)
            return true;

    for (std::size_t j : pending_)
        if (states_[j].status == Status::REST && states_[j].at != MAX_TIME)
            return true;

    return false;
}

// ---------------------------------------------------------------------------
// Sondas: consultam o estado fisiológico do nó sem modificá-lo
// ---------------------------------------------------------------------------

/**
 * O nó pode ser ativado em 'at'? Equivale a:
 *  - DI = at - RT >= limiar (no original, DI < DI_threshold => AT = INF);
 *  - o nó está inativo em 'at' e Node::Activate(at) seria aceito pelo modelo
 *    de APD (o modelo de restituição recusa DIs curtos demais).
 */
template <typename APM, typename CVM>
bool FIMEikonalPropagation<APM, CVM>::CanActivateAt(const Node& node, double at) const {

    if (node.type == CELL_TYPE_VOID)
        return false;

    if (!std::isfinite(at))
        return false;

    const float at_f = static_cast<float>(at);

    // Fonte obsoleta: anterior à última ativação efetivada.
    if (at_f < node.apd_model.getActivationTime())
        return false;

    if (node.apd_model.IsActive(at_f))
        return false;

    if (node.apd_model.getDI(at_f) < params.di_threshold)
        return false;

    APM probe = node.apd_model;
    return probe.Activate(at_f);
}

/**
 * Velocidade de condução longitudinal que o nó teria se fosse ativado com o
 * DI dado. Replica NodeT::ComputeActivation sobre uma cópia do cv_model
 * (em restitution_curve(DI, a, b, c) do original).
 */
template <typename APM, typename CVM>
double FIMEikonalPropagation<APM, CVM>::ProbeConductionVelocity(const Node& node, double di) const {

    CVM cv = node.cv_model;
    cv.Activate(static_cast<float>(di), node.apd_model.getAPD());

    return cv.getConductionVelocity();
}

/**
 * Velocidade do nó na direção dir (unitária), para uma velocidade
 * longitudinal dada.
 *
 * É a mesma elipse de NodeT::ComputeDirectionalConductionVelocity,
 *     v = cv / sqrt(cos^2 + (sin / r)^2),
 * (r = cond_veloc_transversal_reduction, cos = fibra . dir), que é a elipse do
 * original,
 *     1/v^2 = 1/vt^2 + (1/vl^2 - 1/vt^2) df^2,   vt = r vl,
 * mas calculada em double e escrita em função de df^2 (sem sqrt(1 - cos^2)).
 * Não se usa NodeT::ComputeDirectionalConductionVelocity porque, em float,
 * 1 - cos^2 fica levemente negativo quando dir é quase paralela à fibra
 * (p.ex. fibra a 45 graus e vizinho na diagonal), o sqrt dá NaN e o objetivo
 * do eikonal viraria INF no meio da busca em lambda. O original (double)
 * não tem esse problema.
 */
template <typename APM, typename CVM>
double FIMEikonalPropagation<APM, CVM>::DirectionalVelocity(const Node& node, double cond_vel, const Vec3d& dir) const {

    if (node.type == CELL_TYPE_VOID)
        return 0.0;

    const double dir_norm = dir.norm();
    const Vec3d  f        = node.orientation.template cast<double>();

    // Mesmos casos isotrópicos de NodeT::ComputeDirectionalConductionVelocity.
    if (node.parameters->isotropic_diffusion || f.norm() < ALMOST_ZERO || dir_norm < ALMOST_ZERO)
        return cond_vel;

    const double r   = static_cast<double>(node.parameters->cond_veloc_transversal_reduction);
    const double c   = std::fabs(f.dot(dir)) / dir_norm;
    const double df2 = std::min(1.0, c * c);

    const double inv_v2 = (df2 + (1.0 - df2) / (r * r)) / (cond_vel * cond_vel);

    return 1.0 / std::sqrt(inv_v2);
}

// ---------------------------------------------------------------------------
// Eikonal local
// ---------------------------------------------------------------------------

/**
 * objective_function do original: AT do nó para o ponto x(lambda) do segmento
 * entre as fontes (x1, T1) e (x2, T2). Devolve INF se o nó não puder ser
 * ativado nesse AT.
 */
// template <typename APM, typename CVM>
// double FIMEikonalPropagation<APM, CVM>::Objective(Node& node, double lambda, double T1, double T2,
//                                                   const Vec3d& x1, const Vec3d& x2) const {

//     using fim_eikonal_detail::INF;

//     const Vec3d  x  = (1.0 - lambda) * x1 + lambda * x2;
//     const double d  = x.norm();

//     if (!(d > 0.0))
//         return INF;

//     const Vec3d  u  = - x / d;                       // direção de propagação (fonte -> nó)
//     const double T0 = (1.0 - lambda) * T1 + lambda * T2;

//     const double rt_old = static_cast<double>(node.apd_model.getActivationTime()) + static_cast<double>(node.apd_model.getAPD());

//     double AT = INF, old_AT = INF;
//     double DI = params.default_di;

//     for (int iter = 0; iter < params.fixed_point_max_iter; ++iter) {

//         const double cv = ProbeConductionVelocity(node, DI);
//         const double v  = DirectionalVelocity(node, cv, u);

//         if (!(v > 0.0) || !std::isfinite(v))
//             return INF;

//         AT = T0 + d / v;
//         DI = AT - rt_old;

//         if (std::fabs(AT - old_AT) < params.fixed_point_tol)
//             break;

//         old_AT = AT;
//     }

//     return CanActivateAt(node, AT) ? AT : INF;
// }

template <typename APM, typename CVM>
double FIMEikonalPropagation<APM, CVM>::Objective(Node& node, double lambda, double T1, double T2,
                                                  const Vec3d& x1, const Vec3d& x2) const {

    using fim_eikonal_detail::INF;

    const Vec3d  x  = (1.0 - lambda) * x1 + lambda * x2;
    const double d  = x.norm();

    if (!(d > 0.0))
        return INF;

    const Vec3d  u  = - x / d;                            // direção de propagação (fonte -> nó)
    const double T0 = (1.0 - lambda) * T1 + lambda * T2;
    
    const double rt_old = static_cast<double>(node.apd_model.getActivationTime()) + static_cast<double>(node.apd_model.getAPD());

    // // 1. Filtro cinemático do Legacy/FIM para vizinhos na mesma frente de onda:
    // if (node.apd_model.IsActive(static_cast<float>(T0))) {
    //     const double reduction = static_cast<double>(node.parameters->cond_veloc_transversal_reduction);
    //     const double min_vel   = reduction * static_cast<double>(node.conduction_vel);
    //     if (min_vel > 0.0) {
    //         const double earliest_possible = T0 - (d / min_vel);
    //         if (node.id == 4329)
    //         printf("T0 %f AT old %f RT old %f min vel %f d %f earliest %f\n", T0, node.apd_model.getActivationTime(), rt_old, min_vel, d, earliest_possible, node.apd_model.getActivationTime());
    //         if (static_cast<double>(node.apd_model.getActivationTime()) > earliest_possible)
    //             return INF;
    //     }
    // }

    double AT = INF, old_AT = INF;
    double DI = params.default_di;

    // if (node.id == 4329)
    // printf("PONTO FIXO\n");

    for (int iter = 0; iter < params.fixed_point_max_iter; ++iter) {

        const double cv = ProbeConductionVelocity(node, DI);
        const double v  = DirectionalVelocity(node, cv, u);

        if (!(v > 0.0) || !std::isfinite(v))
            return INF;

        double AT_calc = T0 + d / v;
        
        // RELAXAÇÃO NUMÉRICA: Força a convergência cortando a oscilação pela metade
        if (iter > 0) {
            AT = 0.5 * old_AT + 0.5 * AT_calc; 
        } else {
            AT = AT_calc;
        }
        
        DI = AT - rt_old;

        // if (node.id == 4329)
        // printf("T0 %f d %f v %f AT %f RT %f DI %f\n", T0, d, v, AT, rt_old, DI);

        // O Bloqueio de Impacto original (mantém-se)
        if (DI < static_cast<double>(params.di_threshold) || !CanActivateAt(node, AT))
            return INF;

        if (std::fabs(AT - old_AT) < params.fixed_point_tol)
            break; // Convergiu! A onda lateral pode avançar em câmara lenta.

        old_AT = AT;
    }

    return CanActivateAt(node, AT) ? AT : INF;

    // double AT = INF, old_AT = INF;
    // double DI = params.default_di;
    // bool converged = false;

    // if (node.id == 4329)
    // printf("PONTO FIXO\n");

    // for (int iter = 0; iter < params.fixed_point_max_iter; ++iter) {

    //     const double cv = ProbeConductionVelocity(node, DI);
    //     const double v  = DirectionalVelocity(node, cv, u);

    //     if (!(v > 0.0) || !std::isfinite(v))
    //         return INF;

    //     AT = T0 + d / v;
    //     DI = AT - rt_old;

    //     // if (node.id == 4329)
    //     // printf("T0 %f d %f v %f AT %f RT %f DI %f\n", T0, d, v, AT, rt_old, DI);

    //     // 2. A TRAVA PRINCIPAL DE BLOQUEIO:
    //     // Se com a velocidade atual o pulso chegou enquanto a célula ainda está refratária
    //     // (DI < di_threshold ou CanActivateAt falso logo na chegada), bloqueia imediatamente!
    //     // Não permite que a iteração seguinte use DI negativo/baixo para "esticar" d/v e esperar repolarizar.
    //     if (DI < static_cast<double>(params.di_threshold) || !CanActivateAt(node, AT))
    //         return INF;

    //     if (std::fabs(AT - old_AT) < params.fixed_point_tol) {
    //         converged = true;
    //         break;
    //     }

    //     old_AT = AT;
    // }

    // if (!converged) return INF;

    // return CanActivateAt(node, AT) ? AT : INF;
}

/**
 * iterative_eikonal do original: mínimo de Objective em lambda em [0,1] por
 * busca da razão áurea, comparando também os extremos.
 */
template <typename APM, typename CVM>
typename FIMEikonalPropagation<APM, CVM>::Candidate
FIMEikonalPropagation<APM, CVM>::IterativeEikonal(Node& node, double T1, double T2, const Vec3d& x1, const Vec3d& x2) const {

    const double T_left  = Objective(node, 0.0, T1, T2, x1, x2);
    const double T_right = Objective(node, 1.0, T1, T2, x1, x2);

    double lower_lambda = 0.0, upper_lambda = 1.0;
    const double phi = (std::sqrt(5.0) - 1.0) / 2.0;

    double lambda_1 = upper_lambda - phi * (upper_lambda - lower_lambda);
    double lambda_2 = lower_lambda + phi * (upper_lambda - lower_lambda);

    double T_lambda_1 = Objective(node, lambda_1, T1, T2, x1, x2);
    double T_lambda_2 = Objective(node, lambda_2, T1, T2, x1, x2);

    while (std::fabs(upper_lambda - lower_lambda) > params.lambda_tol) {

        if (T_lambda_1 < T_lambda_2) {
            upper_lambda = lambda_2;
            lambda_2     = lambda_1;
            T_lambda_2   = T_lambda_1;
            lambda_1     = upper_lambda - phi * (upper_lambda - lower_lambda);
            T_lambda_1   = Objective(node, lambda_1, T1, T2, x1, x2);
        }
        else {
            lower_lambda = lambda_1;
            lambda_1     = lambda_2;
            T_lambda_1   = T_lambda_2;
            lambda_2     = lower_lambda + phi * (upper_lambda - lower_lambda);
            T_lambda_2   = Objective(node, lambda_2, T1, T2, x1, x2);
        }
    }

    const double lambda = (lower_lambda + upper_lambda) / 2.0;
    const double T_mid  = Objective(node, lambda, T1, T2, x1, x2);

    Candidate c;

    if (T_left <= T_mid && T_left <= T_right) {
        c.at = T_left;  c.lambda = 0.0;
    }
    else if (T_right <= T_mid && T_right <= T_left) {
        c.at = T_right; c.lambda = 1.0;
    }
    else {
        c.at = T_mid;   c.lambda = lambda;
    }

    return c;
}

/**
 * calculate_ta_FIM do original: melhor AT do nó j a partir dos vizinhos que
 * são fontes (têm AT pendente e safety_factor >= o do nó).
 *
 * Fonte = nó com AT tentativo (na lista, convergido ou estímulo). Um nó já
 * efetivado deixa de ser fonte, como no original (RT > AT).
 *
 * 'bound': só interessam candidatos com AT < bound (o chamador descarta os
 * demais, como o original: success = AT < AT_old). Como T >= min(Ta, Tb) em
 * qualquer segmento, segmentos/fontes com tempo >= bound são pulados sem
 * alterar o resultado.
 * Devolve false se não houver candidato finito < bound.
 */
template <typename APM, typename CVM>
bool FIMEikonalPropagation<APM, CVM>::BestCandidate(Tissue& tissue, std::size_t j, double bound, Candidate& best) const {

    using fim_eikonal_detail::INF;

    constexpr size_t K = Geometry::num_neighbours;

    Node& node = tissue.tissue_nodes[j];

    if (node.type == CELL_TYPE_VOID)
        return false;

    const auto& g       = tissue.tissue_geometry;
    const float sf_cell = SafetyFactor(node);

    std::array<bool,   K> is_source;
    std::array<double, K> Ta;
    std::array<int,    K> beat;
    std::array<bool,   K> covered;

    for (size_t k = 0; k < K; ++k) {

        is_source[k] = false;
        covered[k]   = false;
        Ta[k]        = INF;
        beat[k]      = -1;

        Node* n = tissue.NodeDisplace(&node, g.displacement[k]);

        if (n == nullptr || n->type == CELL_TYPE_VOID)
            continue;

        const auto& ns = states_[NodeIndex(tissue, n)];

        if (ns.at == MAX_TIME)
            continue;

        if (SafetyFactor(*n) < sf_cell)
            continue;

        is_source[k] = true;
        Ta[k]        = static_cast<double>(ns.at);
        beat[k]      = ns.beat;
    }

    best.at     = bound;
    best.lambda = 0.0;
    best.beat   = -1;

    bool found = false;

    // Segmentos com as duas pontas fontes.
    for (const auto& e : edges_) {

        const size_t a = e.first, b = e.second;

        if (!is_source[a] || !is_source[b])
            continue;

        covered[a] = covered[b] = true;

        // Sem ganho possível: T >= min(Ta, Tb).
        if (std::min(Ta[a], Ta[b]) >= best.at)
            continue;

        const Candidate c = IterativeEikonal(node, Ta[a], Ta[b], stencil_pos_[a], stencil_pos_[b]);

        if (c.at < best.at) {
            best.at     = c.at;
            best.lambda = c.lambda;
            best.beat   = (c.lambda >= 0.5) ? beat[b] : beat[a];
            found       = true;
        }
    }

    // Fontes sem parceiro-fonte: atualização de um único vizinho (lambda = 1).
    for (size_t k = 0; k < K; ++k) {

        if (!is_source[k] || covered[k])
            continue;

        if (Ta[k] >= best.at)
            continue;

        const double at = Objective(node, 1.0, Ta[k], Ta[k], stencil_pos_[k], stencil_pos_[k]);

        if (at < best.at) {
            best.at     = at;
            best.lambda = 1.0;
            best.beat   = beat[k];
            found       = true;
        }
    }

    // if (node.id == 4329) 
    // printf("Found %d Best AT %f Finite %d\n", found, best.at, std::isfinite(best.at));

    return found && std::isfinite(best.at);
}

// ---------------------------------------------------------------------------
// Lista ativa
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::Enqueue(std::size_t j) {

    if (!states_[j].queued) {
        states_[j].queued = true;
        list_.push_back(j);
    }
}

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::MarkPending(std::size_t j) {

    if (!states_[j].in_pending) {
        states_[j].in_pending = true;
        pending_.push_back(j);
    }
}

template <typename APM, typename CVM>
float FIMEikonalPropagation<APM, CVM>::MinListTime() const {

    float t_min = MAX_TIME;

    for (std::size_t j : list_) {
        const auto& s = states_[j];
        if (s.status == Status::LIST && s.at < t_min)
            t_min = s.at;
    }

    return t_min;
}

// ---------------------------------------------------------------------------
// Estímulo externo (compute_boundary_conditions_FIM)
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::Stimulate(Tissue& tissue, std::size_t node_index, float activation_time, int beat) {

    if (node_index >= tissue.tissue_nodes.size())
        throw std::out_of_range("FIMEikonalPropagation::Stimulate: invalid node index.");

    Node& node = tissue.tissue_nodes[node_index];
    auto& s    = states_[node_index];

    if (node.type == CELL_TYPE_VOID)
        return;

    if (node.GetState(activation_time) != Node::CellActivationState::INACTIVE)
        return;

    if (!(activation_time < s.at))
        return;

    // O legado agenda e depois falha em Activate(); aqui a falha é antecipada
    // para que os vizinhos nunca derivem de uma ativação que não vai ocorrer.
    if (!CanActivateAt(node, activation_time))
        return;

    // O AT do estímulo é fixo (não é recalculado) e o nó sai da lista, se estava.
    s.at     = activation_time;
    s.beat   = beat;
    s.status = Status::REST;
    s.external = true;

    MarkPending(node_index);

    // Os vizinhos só são avaliados em FlushSeeds, depois que todos os nós do
    // estímulo foram marcados (como no original: primeiro AT = TA em todas as
    // células do estímulo, depois active_neighbors).
    if (!s.seed_pending) {
        s.seed_pending = true;
        seeds_.push_back(node_index);
    }

    node.received_potential = 1.0;
}

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::FlushSeeds(Tissue& tissue) {

    if (seeds_.empty())
        return;

    vector<std::size_t> seeds;
    seeds.swap(seeds_);

    for (std::size_t idx : seeds) {

        auto& s = states_[idx];
        s.seed_pending = false;

        if (s.external && s.at != MAX_TIME && s.status == Status::REST)
            ActivateNeighbors(tissue, idx);
    }
}

// ---------------------------------------------------------------------------
// FIM
// ---------------------------------------------------------------------------

/**
 * active_neighbors do original: reavalia os vizinhos em repouso (fora da
 * lista) de um nó convergido. Um vizinho cujo AT melhora volta para a lista.
 */
template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::ActivateNeighbors(Tissue& tissue, std::size_t i) {

    const auto& g = tissue.tissue_geometry;

    Node& src            = tissue.tissue_nodes[i];
    const float sf_src   = SafetyFactor(src);

    for (size_t k = 0; k < Geometry::num_neighbours; ++k) {

        Node* n = tissue.NodeDisplace(&src, g.displacement[k]);

        if (n == nullptr || n->type == CELL_TYPE_VOID)
            continue;

        // if (n->id == 4329) 
        //     printf("Nó %d tentando colocar Nó %d na lista\n", src.id, n->id);

        const std::size_t j = NodeIndex(tissue, n);
        auto& s = states_[j];

        // if (n->id == 4329) 
        //     printf("List %d Ext %d AT Source %f SF %f SF Source %f\n", s.status == Status::LIST, s.external, s.at, SafetyFactor(*n), sf_src);

        if (s.status == Status::LIST)
            continue;

        if (s.external && s.at != MAX_TIME)     // condição de contorno: AT fixo
            continue;

        if (SafetyFactor(*n) > sf_src)
            continue;

        // Só vale se melhorar o AT atual em mais que epsilon (|AT - AT_old| > 1e-5).
        const double bound = (s.at == MAX_TIME) ? fim_eikonal_detail::INF
                                                : static_cast<double>(s.at) - params.epsilon;

        Candidate c;

        if (!BestCandidate(tissue, j, bound, c))
            continue;

        const float at_new = static_cast<float>(c.at);

        // if (n->id == 4329) 
        //     printf("At new %f\n", at_new);

        if (!(at_new < s.at))
            continue;

        // if (n->id == 4329) 
        //     printf("Nó %d colocando Nó %d na lista com AT %f\n", src.id, n->id, at_new);
        
        s.at     = at_new;
        s.beat   = c.beat;
        s.status = Status::LIST;
        Enqueue(j);
    }
}

/**
 * Laço interno de FIM(): varreduras da lista até o menor AT da lista passar
 * de t_start + window.
 */
template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::RelaxWindow(Tissue& tissue, float t_start) {

    const double window_end = static_cast<double>(t_start) + static_cast<double>(params.window);

    size_t sweeps = 0;

    do {

        if (++sweeps > params.max_sweeps)
            throw std::runtime_error("FIMEikonalPropagation::RelaxWindow: too many sweeps (no convergence).");

        vector<std::size_t> current;
        current.swap(list_);

        for (std::size_t j : current) {

            auto& s = states_[j];
            s.queued = false;

            if (s.status != Status::LIST)
                continue;       // entrada obsoleta (ex.: virou estímulo)

            const float at_old = s.at;

            Candidate c;

            if (BestCandidate(tissue, j, static_cast<double>(s.at), c)) {

                const float at_new = static_cast<float>(c.at);

                if (at_new < s.at) {
                    s.at   = at_new;
                    s.beat = c.beat;
                }
            }

            if (std::fabs(static_cast<double>(s.at) - static_cast<double>(at_old)) < params.epsilon) {

                if (static_cast<double>(s.at) <= window_end) {

                    // Convergiu: deixa a lista, passa a aguardar efetivação e
                    // reavalia os vizinhos em repouso.
                    s.status = Status::REST;
                    MarkPending(j);
                    ActivateNeighbors(tissue, j);
                }
                else {
                    Enqueue(j);     // fora da janela: espera
                }
            }
            else {
                Enqueue(j);
            }
        }

    } while (MinListTime() < t_start + params.window);
}

// ---------------------------------------------------------------------------
// Efetivação
// ---------------------------------------------------------------------------

/**
 * compute_apd do original: efetiva (NodeT::Activate) os nós convergidos cujo
 * AT é definitivo (AT <= menor AT da lista - tau_s) e está dentro do limite.
 */
template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::CommitFinalized(Tissue& tissue, float t_list, float limit, bool inclusive) {

    const double eligible = (t_list == MAX_TIME)
        ? fim_eikonal_detail::INF
        : static_cast<double>(t_list) - static_cast<double>(params.tau_s);

    vector<std::pair<float, std::size_t>> due;
    vector<std::size_t> keep;

    for (std::size_t j : pending_) {

        auto& s = states_[j];

        if (s.status != Status::REST || s.at == MAX_TIME) {
            s.in_pending = false;       // entrada obsoleta
            continue;
        }

        const bool in_limit = inclusive ? (s.at <= limit) : (s.at < limit);

        if (in_limit && static_cast<double>(s.at) <= eligible)
            due.emplace_back(s.at, j);
        else
            keep.push_back(j);
    }

    pending_.swap(keep);

    std::sort(due.begin(), due.end());

    for (const auto& item : due) {

        states_[item.second].in_pending = false;

        // Recuperações anteriores a esta ativação, para manter a ordem temporal.
        ProcessRecoveries(tissue, item.first, false);

        CommitNode(tissue, item.second);
    }
}

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::CommitNode(Tissue& tissue, std::size_t idx) {

    Node& node = tissue.tissue_nodes[idx];
    auto& s    = states_[idx];

    const float at   = s.at;
    const int   beat = s.beat;

    // O nó deixa de ter ativação pendente (e de ser fonte).
    s.at           = MAX_TIME;
    s.beat         = -1;
    s.external     = false;
    s.status       = Status::REST;

    if (node.recovery_time != MAX_TIME && node.recovery_time <= at)
        ApplyRecovery(tissue, idx);

    if (at > tissue.tissue_time)
        tissue.tissue_time = at;

    node.received_potential = 1.0;

    if (node.Activate(at, beat, &tissue)) {
        tissue.apd_variation += node.apd_model.getDeltaAPD();
        recovery_heap_.emplace(node.recovery_time, idx);
    }

    if (node.parameters != nullptr && node.parameters->sensor)
        tissue.sensor_dict.AddData(node.id, node.GetData(at));
}

// ---------------------------------------------------------------------------
// Recuperação
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::ApplyRecovery(Tissue& tissue, std::size_t idx) {

    Node& node = tissue.tissue_nodes[idx];

    if (node.recovery_time == MAX_TIME)
        return;

    const float recovery_time = node.recovery_time;

    node.received_potential = 0.0;
    node.recovery_time      = MAX_TIME;

    if (node.parameters != nullptr && node.parameters->sensor)
        tissue.sensor_dict.AddData(node.id, node.GetData(recovery_time));
}

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::ProcessRecoveries(Tissue& tissue, float limit, bool inclusive) {

    while (!recovery_heap_.empty()) {

        const RecoveryEntry top = recovery_heap_.top();

        const bool due = inclusive ? (top.first <= limit) : (top.first < limit);

        if (!due)
            break;

        recovery_heap_.pop();

        Node& node = tissue.tissue_nodes[top.second];

        if (node.recovery_time != top.first)
            continue;       // entrada obsoleta

        ApplyRecovery(tissue, top.second);

        if (top.first > tissue.tissue_time)
            tissue.tissue_time = top.first;
    }
}

// ---------------------------------------------------------------------------
// AdvanceUntil
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::AdvanceUntil(Tissue& tissue, float time_limit, bool inclusive) {

    if (time_limit < tissue.GetTime())
        throw std::runtime_error("Cannot advance propagation backwards.");

    CheckSupported(tissue);

    auto within_limit = [&](double t) {
        return inclusive ? (t <= time_limit) : (t < time_limit);
    };

    FlushSeeds(tissue);

    while (true) {

        float t_list = MinListTime();

        // Um AT só é definitivo quando AT <= t_list - tau_s. Para efetivar tudo
        // o que cai dentro do limite é preciso relaxar até t_list - tau_s passar
        // do limite. Nunca se efetiva nada além do limite.
        const bool relax = (t_list != MAX_TIME) &&
                           within_limit(static_cast<double>(t_list) - static_cast<double>(params.tau_s));

        if (relax) {
            RelaxWindow(tissue, t_list);
            t_list = MinListTime();
        }

        CommitFinalized(tissue, t_list, time_limit, inclusive);

        if (!relax)
            break;
    }

    ProcessRecoveries(tissue, time_limit, inclusive);
}

// ---------------------------------------------------------------------------
// Persistência
// ---------------------------------------------------------------------------

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::SaveState(std::ofstream& f, const Tissue& tissue) const {

    constexpr int solver_id     = static_cast<int>(PropagationSolverType::FIMEikonal);
    constexpr int state_version = 1;

    f.write(reinterpret_cast<const char*>(&solver_id), sizeof(solver_id));
    f.write(reinterpret_cast<const char*>(&state_version), sizeof(state_version));

    if (!f)
        throw std::runtime_error("FIMEikonalPropagation::SaveState: failed to write header.");

    const std::size_t n_states = states_.size();

    if (n_states != tissue.tissue_nodes.size())
        throw std::runtime_error("FIMEikonalPropagation::SaveState: node count mismatch.");

    f.write(reinterpret_cast<const char*>(&n_states), sizeof(n_states));

    for (const auto& s : states_) {

        const unsigned char status = static_cast<unsigned char>(s.status);

        f.write(reinterpret_cast<const char*>(&s.at), sizeof(s.at));
        f.write(reinterpret_cast<const char*>(&s.beat), sizeof(s.beat));
        f.write(reinterpret_cast<const char*>(&s.external), sizeof(s.external));
        f.write(reinterpret_cast<const char*>(&s.seed_pending), sizeof(s.seed_pending));
        f.write(reinterpret_cast<const char*>(&status), sizeof(status));
    }

    if (!f)
        throw std::runtime_error("FIMEikonalPropagation::SaveState: failed to write propagation state.");
}

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::LoadState(std::ifstream& f, Tissue& tissue) {

    int solver_id     = -1;
    int state_version = -1;

    f.read(reinterpret_cast<char*>(&solver_id), sizeof(solver_id));
    f.read(reinterpret_cast<char*>(&state_version), sizeof(state_version));

    const int expected_id = static_cast<int>(PropagationSolverType::FIMEikonal);

    if (!f || solver_id != expected_id)
        throw std::runtime_error("FIMEikonalPropagation::LoadState: solver type mismatch.");

    if (state_version != 1)
        throw std::runtime_error("FIMEikonalPropagation::LoadState: unsupported state version.");

    std::size_t n_states = 0;
    f.read(reinterpret_cast<char*>(&n_states), sizeof(n_states));

    if (!f || n_states != tissue.tissue_nodes.size())
        throw std::runtime_error("FIMEikonalPropagation::LoadState: checkpoint node count mismatch or truncated state.");

    states_.assign(n_states, PropagationState{});

    for (std::size_t i = 0; i < n_states; ++i) {

        auto& s = states_[i];
        unsigned char status = 0;

        f.read(reinterpret_cast<char*>(&s.at), sizeof(s.at));
        f.read(reinterpret_cast<char*>(&s.beat), sizeof(s.beat));
        f.read(reinterpret_cast<char*>(&s.external), sizeof(s.external));
        f.read(reinterpret_cast<char*>(&s.seed_pending), sizeof(s.seed_pending));
        f.read(reinterpret_cast<char*>(&status), sizeof(status));

        if (!f)
            throw std::runtime_error("FIMEikonalPropagation::LoadState: truncated propagation state.");

        if (status > static_cast<unsigned char>(Status::LIST))
            throw std::runtime_error("FIMEikonalPropagation::LoadState: invalid node status.");

        s.status = static_cast<Status>(status);
    }

    RebuildQueues(tissue);
}

template <typename APM, typename CVM>
void FIMEikonalPropagation<APM, CVM>::RebuildQueues(Tissue& tissue) {

    list_.clear();
    pending_.clear();
    seeds_.clear();
    recovery_heap_ = RecoveryHeap();

    for (std::size_t i = 0; i < states_.size(); ++i) {

        auto& s = states_[i];
        s.queued     = false;
        s.in_pending = false;

        if (s.status == Status::LIST)
            Enqueue(i);
        else if (s.at != MAX_TIME)
            MarkPending(i);

        if (s.seed_pending)
            seeds_.push_back(i);

        if (tissue.tissue_nodes[i].recovery_time != MAX_TIME)
            recovery_heap_.emplace(tissue.tissue_nodes[i].recovery_time, i);
    }
}

#endif // FIM_EIKONAL_PROPAGATION_IMPL_H
