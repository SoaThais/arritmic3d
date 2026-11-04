
import meshio
import numpy as np
import sys


def compare_vtu(reference_file, test_file, top_n=10):
    ref = meshio.read(reference_file)
    test = meshio.read(test_file)

    print("=" * 100)
    print("COMPARAÇÃO DOS CAMPOS FINAIS")
    print("=" * 100)

    # Verifica se os arquivos possuem a mesma malha
    if ref.points.shape != test.points.shape:
        raise ValueError(
            f"Número de pontos diferente: "
            f"REF={ref.points.shape}, TEST={test.points.shape}"
        )

    if not np.allclose(ref.points, test.points, rtol=0, atol=1e-8):
        raise ValueError(
            "As coordenadas dos nós são diferentes. "
            "Não é seguro comparar os campos pelo índice."
        )

    print(f"Quantidade de nós: {len(ref.points)}")
    print(f"Campos REF: {list(ref.point_data.keys())}")
    print(f"Campos TEST: {list(test.point_data.keys())}")

    common_fields = sorted(
        set(ref.point_data.keys()) & set(test.point_data.keys())
    )

    only_ref = sorted(set(ref.point_data.keys()) - set(test.point_data.keys()))
    only_test = sorted(set(test.point_data.keys()) - set(ref.point_data.keys()))

    if only_ref:
        print("\nCampos ausentes no TEST:", only_ref)
    if only_test:
        print("\nCampos ausentes na REF:", only_test)

    print("\n" + "=" * 100)
    print("RESUMO GLOBAL")
    print("=" * 100)

    print(
        f"{'Campo':<15}"
        f"{'Tipo':<12}"
        f"{'Nós dif.':>12}"
        f"{'% dif.':>10}"
        f"{'Erro máx.':>16}"
        f"{'MAE':>16}"
        f"{'RMSE':>16}"
    )
    print("-" * 100)

    for field in common_fields:
        a = np.asarray(ref.point_data[field])
        b = np.asarray(test.point_data[field])

        if a.shape != b.shape:
            print(
                f"{field:<15} SHAPE DIFERENTE: "
                f"REF={a.shape}, TEST={b.shape}"
            )
            continue

        # Campos numéricos
        if (
            np.issubdtype(a.dtype, np.number)
            and np.issubdtype(b.dtype, np.number)
        ):
            a_float = a.astype(np.float64)
            b_float = b.astype(np.float64)

            finite = np.isfinite(a_float) & np.isfinite(b_float)
            if a.ndim > 1:
                finite_nodes = np.all(finite.reshape(len(a), -1), axis=1)
                errors = np.abs(b_float - a_float)
                node_errors = np.max(
                    errors.reshape(len(a), -1), axis=1
                )
            else:
                finite_nodes = finite
                node_errors = np.abs(b_float - a_float)

            valid_errors = node_errors[finite_nodes]

            if len(valid_errors) == 0:
                print(f"{field:<15} SEM VALORES FINITOS COMPARÁVEIS")
                continue

            different = finite_nodes & (node_errors > 0)
            n_diff = int(np.count_nonzero(different))
            pct_diff = 100.0 * n_diff / len(a)

            max_error = float(np.max(valid_errors))
            mae = float(np.mean(valid_errors))
            rmse = float(np.sqrt(np.mean(node_errors[finite_nodes] ** 2)))

            if (
                np.issubdtype(a.dtype, np.integer)
                and np.issubdtype(b.dtype, np.integer)
            ):
                field_type = "inteiro"
            else:
                field_type = "numérico"

            print(
                f"{field:<15}"
                f"{field_type:<12}"
                f"{n_diff:>12}"
                f"{pct_diff:>9.4f}%"
                f"{max_error:>16.8g}"
                f"{mae:>16.8g}"
                f"{rmse:>16.8g}"
            )

            # Mostra os nós com maiores diferenças
            if n_diff > 0:
                
                candidates = np.flatnonzero(different)
                
                # --- INÍCIO DA BUSCA PELO MENOR LAT ---
                if 'LAT' in ref.point_data:
                    lat_ref = np.asarray(ref.point_data['LAT'])
                    
                    # Pega os valores de LAT apenas dos nós que deram diferença neste campo
                    lat_of_candidates = lat_ref[candidates]
                    
                    # Encontra o nó com o menor LAT entre os que divergiram
                    idx_min_lat = np.argmin(lat_of_candidates)
                    primeiro_no_tempo = candidates[idx_min_lat]
                    
                    print(f"\n  -> ⏱️ Primeiro nó a divergir no tempo (menor LAT na REF): {primeiro_no_tempo}")
                    print(f"     LAT neste nó: {lat_ref[primeiro_no_tempo]}")
                    print(f"     Valor '{field}' -> REF: {a[primeiro_no_tempo]} | TEST: {b[primeiro_no_tempo]} | Erro: {node_errors[primeiro_no_tempo]:.8g}")
                # --- FIM DA BUSCA PELO MENOR LAT ---

                order = candidates[
                    np.argsort(node_errors[candidates])[::-1]
                ][:top_n]

                print(f"\n  Maiores diferenças em '{field}':")
                print(
                    f"  {'Nó':>10} {'REF':>18} "
                    f"{'TEST':>18} {'Erro abs.':>18}"
                )

                for node in order:
                    print(
                        f"  {node:>10}"
                        f" {str(a[node]):>18}"
                        f" {str(b[node]):>18}"
                        f" {node_errors[node]:>18.8g}"
                    )

                print()

        else:
            # Campos não numéricos
            equal = np.all(a == b, axis=tuple(range(1, a.ndim))) \
                if a.ndim > 1 else (a == b)
            n_diff = int(np.count_nonzero(~equal))
            print(
                f"{field:<15}{'não numérico':<12}"
                f"{n_diff:>12}"
                f"{100*n_diff/len(a):>9.4f}%"
            )

    print("=" * 100)
    print("FIM DA COMPARAÇÃO")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print(
            "Uso:\n"
            "python3 compare_vtu.py reference.vtu test.vtu [top_n]"
        )
        sys.exit(1)

    reference_file = sys.argv[1]
    test_file = sys.argv[2]
    top_n = int(sys.argv[3]) if len(sys.argv) == 4 else 10

    compare_vtu(reference_file, test_file, top_n)
