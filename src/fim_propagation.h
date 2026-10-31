#ifndef FIM_PROPAGATION_H
#define FIM_PROPAGATION_H

#include <vector>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <algorithm>

#include "geometry.h"
#include "front_propagation_solver.h"

using std::vector;

/**
 * @brief Propagação baseada no Fast Iterative Method (FIM).
 *
 * Substitui a fila de eventos (heap) do LegacyHeapPropagation por uma lista
 * ativa não ordenada, no estilo do FIM com janela temporal usado no
 * autômato celular eikonal (AutomatoCelularCardiaco):
 *
 *  - Cada nó mantém um tempo de ativação TENTATIVO (tentative_at) da sua
 *    próxima ativação, sem alterar o estado fisiológico do NodeT.
 *  - Nós com valor tentativo ainda sujeito a mudança ficam na lista ativa
 *    (Status::LIST). Em cada varredura, um nó da lista recalcula o AT
 *    puxando dos vizinhos já estáveis (BestCandidate) e compara com o valor
 *    anterior; se não mudou, converge (Status::CONVERGED), sai da lista,
 *    passa a valer como fonte e seus vizinhos são reavaliados
 *    (ActivateNeighbors); um vizinho cujo valor melhora volta para a lista
 *    (inclusive nós já convergidos). Nós ainda na lista nunca propagam.
 *  - A relaxação é feita por janelas de largura `window` (dt do FIM
 *    original) a partir do menor valor tentativo da lista.
 *  - Um nó convergido só é EFETIVADO (NodeT::Activate: APD, CV, tempo de
 *    recuperação, sensores) quando seu valor é definitivo, isto é, quando
 *    tentative_at <= (menor valor na lista) - tau_s, e dentro do limite
 *    pedido em AdvanceUntil. As efetivações ocorrem em ordem crescente de
 *    tempo.
 *
 * Como a velocidade de condução de um nó depende do seu DI na ativação, e
 * isso só é conhecido ao efetivar, a velocidade usada para propagar a partir
 * de um nó ainda tentativo é obtida por uma sonda: cópia do cv_model do nó
 * ativada com o DI/APD que ele teria em tentative_at (o nó não é alterado).
 *
 * Regras preservadas do solver legado:
 *  - T_j = T_i + d_ij / c_ij, com c_ij a velocidade direcional do nó i;
 *  - safety factor: pai com safety_factor menor que o do filho não propaga;
 *  - refratariedade: ativação só é aceita se o nó estiver inativo em T_j e
 *    o modelo de APD aceitar a ativação (equivale ao Activate() do legado).
 *
 * Limitações (levantam exceção, para não produzir resultado silenciosamente
 * diferente do legado):
 *  - long_apd_reactivation;
 *  - min_potential > 0 (mecanismo de potencial recebido).
 *  received_potential é mantido apenas de forma informativa.
 */
template <typename APM, typename CVM>
class FIMPropagation : public FrontPropagationSolver<CardiacTissue<APM, CVM>> {

    public:

        using Tissue = CardiacTissue<APM, CVM>;
        using Node   = NodeT<APM, CVM>;

        enum class Status : unsigned char {
            REST      = 0,   ///< Sem ativação tentativa pendente.
            LIST      = 1,   ///< Na lista ativa: valor ainda pode mudar.
            CONVERGED = 2    ///< Valor estável, aguardando efetivação.
        };

        struct Parameters {
            float  window     = 20.0f;   ///< Largura da janela de relaxação (dt do FIM), em ms.
            float  tau_s      = 0.0f;    ///< Margem de segurança para efetivar (ms). 0 = critério exato.
            float  epsilon    = 1e-4f;   ///< Melhora mínima para atualizar um valor tentativo (ms).
            size_t max_sweeps = 10000000;///< Proteção contra laço infinito na relaxação.
        };

        struct PropagationState {
            float tentative_at = MAX_TIME;
            float velocity     = 0.0f;
            std::int64_t parent= -1;
            int   beat         = -1;
            bool  external     = false;
            bool  queued       = false;
            Status status      = Status::REST;
            
            float source_at    = MAX_TIME;
            float source_vel   = 0.0f;
            int   source_beat  = -1;

            // --- NOVAS VARIÁVEIS DE POTENCIAL ---
            float tentative_potential = 0.0f; // Acumulado durante a iteração
            float source_potential    = 0.0f; // Total final consolidado
            float outgoing_potential  = 0.0f; // Fração que será repassada aos sumidouros
        };

        struct Candidate {
            float        at       = MAX_TIME;
            std::int64_t parent   = -1;
            int          beat     = -1;
            float        potential = 0.0f; // Potencial que esse candidato acumulou
        };

        FIMPropagation() = default;

        PropagationSolverType Type() const override {
            return PropagationSolverType::FIM;
        }

        void Initialize(Tissue& tissue) override;

        bool HasPendingWork() const override;

        void AdvanceUntil(Tissue& tissue, float time_limit, bool inclusive) override;

        void Stimulate(Tissue& tissue, std::size_t node_index, float activation_time, int beat) override;

        void SaveState(std::ofstream& f, const Tissue& tissue) const override;
        void LoadState(std::ifstream& f, Tissue& tissue) override;

        void SetParameters(const Parameters& p) { params = p; }
        const Parameters& GetParameters() const { return params; }

    private:

        Parameters               params;
        vector<PropagationState> states;
        vector<std::size_t>      list_;             ///< Lista ativa (pode conter entradas obsoletas).
        vector<std::size_t>      converged_;        ///< Nós convergidos aguardando efetivação (idem).
        vector<std::size_t>      recovery_queue_;   ///< Nós com recuperação pendente (idem).

        // ---- suporte ----
        void RebuildQueues(Tissue& tissue);
        std::size_t NodeIndex(const Tissue& tissue, const Node* node) const;

        // ---- sondas (não alteram o nó) ----
        bool  CanActivateAt(const Node& node, float at) const;
        float ProbeConductionVelocity(const Node& node, float at) const;
        float DirectionalVelocity(Node& node, float cond_vel, const typename Node::Vector3& dir) const;

        // ---- lista ativa ----
        void  Enqueue(std::size_t j);
        void  SetTentative(Tissue& tissue, std::size_t j, float at, std::int64_t parent, int beat, bool external, double potential);
        void  ClearTentative(std::size_t j);
        float MinListTime() const;

        // ---- FIM ----
        void RelaxWindow(Tissue& tissue, float t_start);
        bool BestCandidate(Tissue& tissue, std::size_t j, Candidate& best);
        void ActivateNeighbors(Tissue& tissue, std::size_t i);

        // ---- efetivação ----
        void CommitFinalized(Tissue& tissue, float t_list, float limit, bool inclusive);
        void CommitNode(Tissue& tissue, std::size_t idx);

        // ---- recuperação ----
        void ApplyRecovery(Tissue& tissue, std::size_t idx);
        void ProcessRecoveries(Tissue& tissue, float limit, bool inclusive);

};

#include "fim_propagation_impl.h"

#endif // FIM_PROPAGATION_H