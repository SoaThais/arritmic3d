#ifndef FIM_EIKONAL_PROPAGATION_H
#define FIM_EIKONAL_PROPAGATION_H

#include <vector>
#include <queue>
#include <array>
#include <utility>
#include <functional>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#include "geometry.h"
#include "front_propagation_solver.h"

using std::vector;

/**
 * @brief FIM eikonal (Fast Iterative Method) dentro da estrutura do Arritmic3D.
 *
 * Esta classe NÃO tem relação com FIMPropagation. Ela é um porte do FIM
 * original do autômato celular eikonal (calculate_ta_FIM / iterative_eikonal /
 * objective_function / active_neighbors / FIM) para a interface
 * FrontPropagationSolver do Arritmic3D. A lógica numérica é a do código
 * original; o que muda é de onde vêm os dados:
 *
 *   FIM original                         Arritmic3D (aqui)
 *   -----------------------------------  ---------------------------------------
 *   cell->activation_time (AT)           PropagationState::at (AT tentativo)
 *   cell->deactivation_time (RT)         apd_model.getActivationTime()+getAPD()
 *   restitution_curve(DI, a, b, c)       cv_model.Activate(DI, APD) (sonda, sem alterar o nó)
 *   vl, vt (elipse de velocidade)        NodeT::ComputeDirectionalConductionVelocity
 *   DI_threshold                         aceitação do modelo de APD (+ di_threshold opcional)
 *   compute_apd (RT = AT + APD)          NodeT::Activate (efetivação)
 *   states[] (0 = repouso, 1 = lista)    PropagationState::status
 *   anel de 8 vizinhos (2D)              arestas da casca de vizinhos (3D, ver abaixo)
 *
 * Algoritmo (idêntico ao original):
 *
 *  1. Atualização local: o AT de um nó j é o mínimo, sobre "triângulos"
 *     (j, vizinho a, vizinho b) cujos dois vizinhos são fontes, do tempo
 *         T(lambda) = T0(lambda) + d(lambda) / v(DI(T)),
 *     com x(lambda) = (1-lambda) x_a + lambda x_b, T0 = (1-lambda) T_a + lambda T_b,
 *     d = |x| e v a velocidade direcional do próprio nó j ao longo de -x/|x|.
 *     O mínimo em lambda em [0,1] é obtido por busca da razão áurea (tol. 1e-3),
 *     comparando também os extremos. Como v depende do DI = T - RT_j, e T
 *     depende de v, cada avaliação resolve o ponto fixo T = T0 + d / v(T - RT_j)
 *     (até 100 iterações, tol. 1e-5, começando em DI = default_di).
 *     Um vizinho fonte sem parceiro-fonte usa a atualização de um único
 *     vizinho (lambda = 1). Candidato com DI abaixo do limiar vale INF.
 *
 *  2. Lista ativa com janela: nós com AT melhorado entram na lista. Cada
 *     varredura recalcula todos os nós da lista; um nó cujo AT não mudou
 *     (|dAT| < epsilon) e que está dentro da janela [t0, t0 + window]
 *     CONVERGE: sai da lista e reavalia os vizinhos em repouso
 *     (active_neighbors), que voltam à lista se melhoram. A janela termina
 *     quando o menor AT da lista passa de t0 + window.
 *
 *  3. Efetivação (compute_apd): um nó convergido com AT <= (menor AT da
 *     lista) - tau_s é definitivo e é efetivado com NodeT::Activate (APD, CV,
 *     recovery_time, sensores), em ordem crescente de tempo e respeitando o
 *     limite de AdvanceUntil. Depois de efetivado o nó deixa de ser fonte
 *     (no original: RT > AT).
 *
 * Geometria 3D. O original usa o anel de 8 vizinhos 2D, onde segmentos são
 * pares consecutivos do anel. Aqui os segmentos são as arestas unitárias da
 * casca de cada cubo de vizinhança: pares de vizinhos que diferem de 1 passo
 * da grade em um único eixo e estão na mesma casca (Chebyshev). Para
 * NEIGHBOURS_DISTANCE = 1 são 48 arestas (em um slab 2D reduz exatamente ao
 * anel de 8 vizinhos do original).
 *
 * Preservado do solver legado: safety_factor (a fonte precisa ter
 * safety_factor >= o do alvo), refratariedade (via apd_model), atualização
 * de sensores e do recovery_time.
 *
 * Não implementado no original e, portanto, aqui (lançam exceção em vez de
 * dar resultado silenciosamente diferente do legado):
 *  - min_potential > 0 (mecanismo de potencial recebido);
 *  - long_apd_reactivation;
 *  - correção de curvatura (kappa = 0, como na chamada de FIM() original com
 *    curvature_update = false).
 */
template <typename APM, typename CVM>
class FIMEikonalPropagation : public FrontPropagationSolver<CardiacTissue<APM, CVM>> {

    public:

        using Tissue = CardiacTissue<APM, CVM>;
        using Node   = NodeT<APM, CVM>;
        using Vec3d  = Eigen::Vector3d;

        enum class Status : unsigned char {
            REST = 0,   ///< Fora da lista (sem AT tentativo, ou AT convergido aguardando efetivação).
            LIST = 1    ///< Na lista ativa: o AT ainda pode mudar.
        };

        struct Parameters {
            float  window               = 20.0f;    ///< dt do FIM original (ms): largura da janela de relaxação.
            float  tau_s                = 0.0f;    ///< tau_s do original (ms): margem para efetivar um AT.
            float  epsilon              = 1e-5f;    ///< Tolerância de convergência do AT (ms).
            float  di_threshold         = 0.0f;     ///< Limiar de DI adicional ao do modelo de APD (ms).
            float  default_di           = 1000.0f;  ///< DI inicial do ponto fixo (default_value do original).
            double fixed_point_tol      = 1e-5;     ///< tol do ponto fixo T = T0 + d/v(DI(T)).
            int    fixed_point_max_iter = 100;      ///< num_iter do original.
            double lambda_tol           = 1e-3;     ///< tol da busca da razão áurea.
            size_t max_sweeps           = 10000000; ///< Proteção contra laço infinito.
        };

        struct PropagationState {
            float  at           = MAX_TIME;         ///< AT tentativo (MAX_TIME = sem ativação pendente).
            int    beat         = -1;               ///< Beat da ativação pendente.
            bool   external     = false;            ///< AT fixo por estímulo externo (condição de contorno).
            bool   seed_pending = false;            ///< Estímulo ainda não propagado aos vizinhos.
            Status status       = Status::REST;
            bool   queued       = false;            ///< Em list_ (não persistido).
            bool   in_pending   = false;            ///< Em pending_ (não persistido).
        };

        struct Candidate {
            double at     = 0.0;
            double lambda = 0.0;
            int    beat   = -1;
        };

        FIMEikonalPropagation() = default;

        PropagationSolverType Type() const override {
            return PropagationSolverType::FIMEikonal;
        }

        void Initialize(Tissue& tissue) override;

        bool HasPendingWork() const override;

        void AdvanceUntil(Tissue& tissue, float time_limit, bool inclusive) override;

        void Stimulate(Tissue& tissue, std::size_t node_index, float activation_time, int beat) override;

        void SaveState(std::ofstream& f, const Tissue& tissue) const override;
        void LoadState(std::ifstream& f, Tissue& tissue) override;

        void SetParameters(const Parameters& p) { params = p; }
        const Parameters& GetParameters() const { return params; }

        /// Número de arestas do estêncil (diagnóstico/testes).
        std::size_t NumStencilEdges() const { return edges_.size(); }

    private:

        using RecoveryEntry = std::pair<float, std::size_t>;
        using RecoveryHeap  = std::priority_queue<RecoveryEntry, vector<RecoveryEntry>, std::greater<RecoveryEntry>>;

        Parameters                       params;
        vector<PropagationState>         states_;
        vector<std::size_t>              list_;          ///< Lista ativa.
        vector<std::size_t>              pending_;       ///< Nós convergidos aguardando efetivação.
        vector<std::size_t>              seeds_;         ///< Estímulos a propagar.
        RecoveryHeap                     recovery_heap_; ///< Recuperações pendentes (menor tempo primeiro).

        // Estêncil (construído em Initialize)
        vector<Vec3d>                                 stencil_pos_;   ///< Posição (física) de cada vizinho em relação ao nó.
        vector<std::pair<std::size_t, std::size_t>>   edges_;         ///< Pares de vizinhos que formam os segmentos.

        // ---- suporte ----
        void        BuildStencil(Tissue& tissue);
        void        CheckSupported(const Tissue& tissue) const;
        std::size_t NodeIndex(const Tissue& tissue, const Node* node) const;
        static float SafetyFactor(const Node& node);

        // ---- sondas (não alteram o estado fisiológico do nó) ----
        bool   CanActivateAt(const Node& node, double at) const;
        double ProbeConductionVelocity(const Node& node, double di) const;
        double DirectionalVelocity(const Node& node, double cond_vel, const Vec3d& dir) const;

        // ---- eikonal local (objective_function / iterative_eikonal) ----
        double    Objective(Node& node, double lambda, double T1, double T2, const Vec3d& x1, const Vec3d& x2) const;
        Candidate IterativeEikonal(Node& node, double T1, double T2, const Vec3d& x1, const Vec3d& x2) const;
        bool      BestCandidate(Tissue& tissue, std::size_t j, double bound, Candidate& best) const;

        // ---- lista ativa ----
        void  Enqueue(std::size_t j);
        void  MarkPending(std::size_t j);
        float MinListTime() const;

        // ---- FIM ----
        void FlushSeeds(Tissue& tissue);
        void ActivateNeighbors(Tissue& tissue, std::size_t i);
        void RelaxWindow(Tissue& tissue, float t_start);

        // ---- efetivação / recuperação ----
        void CommitFinalized(Tissue& tissue, float t_list, float limit, bool inclusive);
        void CommitNode(Tissue& tissue, std::size_t idx);
        void ApplyRecovery(Tissue& tissue, std::size_t idx);
        void ProcessRecoveries(Tissue& tissue, float limit, bool inclusive);
        void RebuildQueues(Tissue& tissue);

};

#include "fim_eikonal_propagation_impl.h"

#endif // FIM_EIKONAL_PROPAGATION_H
