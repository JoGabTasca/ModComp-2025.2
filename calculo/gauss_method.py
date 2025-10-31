import numpy as np

def gauss_somente(A, b, mostrar_passos=True, cond_limite=1e5):
    """
    Resolve sistemas lineares Ax = b APENAS por Eliminação de Gauss
    quando matematicamente possível. Para outros casos, exibe aviso
    e oferece continuar com outro método.
    """

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    m, n = A.shape

    # CASO 1️: MATRIZ QUADRADA
    if m == n:
        # Verifica se a matriz é singular
        if np.linalg.matrix_rank(A) < n:
            print("\n⚠️ Este sistema é SINGULAR (det(A)=0).")
            print("Esse problema NÃO pode ser resolvido por Eliminação de Gauss.")
            print("Método sugerido: Decomposição em Valores Singulares (SVD).")
            opcao = input("Deseja continuar usando o método SVD? (s/n): ").strip().lower()
            if opcao == 's':
                return resolver_por_svd(A, b)
            else:
                print("Operação encerrada. Use um sistema que possa ser resolvido por Gauss.")
                return None
        else:
            # Verifica o número de condição
            cond = np.linalg.cond(A)
            if cond > cond_limite:
                print(f"⚠️ Sistema mal condicionado (cond(A) = {cond:.2e}).")
                print("O resultado da Eliminação de Gauss pode conter erros numéricos.")
            else:
                print(f"✅ Sistema bem condicionado (cond(A) = {cond:.2f}).")

            print("\n✅ Sistema quadrado e não singular — aplicando Eliminação de Gauss.")
            return eliminacao_de_gauss_completa(A, b, mostrar_passos)

    # CASO 2️: MATRIZ NÃO QUADRADA
    else:
        print("\n⚠️ Este sistema NÃO é quadrado (m ≠ n).")
        print("Esse problema NÃO pode ser resolvido por Eliminação de Gauss.")
        print("Método sugerido: Mínimos Quadrados (Normal Equations).")
        opcao = input("Deseja continuar com o método dos Mínimos Quadrados? (s/n): ").strip().lower()
        if opcao == 's':
            return resolver_por_minimos_quadrados(A, b)
        else:
            print("Operação encerrada. Use um sistema quadrado para aplicar Eliminação de Gauss.")
            return None


def eliminacao_de_gauss_completa(A, b, mostrar_passos=False):
    """
    🔹 Implementação da Eliminação de Gauss com pivoteamento total.
    (Usada apenas em sistemas quadrados e não singulares)
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    n = len(b)
    Ab = np.hstack([A, b])

    if mostrar_passos:
        print("\nMatriz aumentada inicial:")
        print(Ab, "\n")

    # --- Etapa de eliminação progressiva ---
    for k in range(n - 1):
        sub_matriz = abs(Ab[k:n, k:n])
        i_max, j_max = np.unravel_index(np.argmax(sub_matriz), sub_matriz.shape)
        i_max += k
        j_max += k

        # Troca de linhas e colunas
        Ab[[k, i_max]] = Ab[[i_max, k]]
        Ab[:, [k, j_max]] = Ab[:, [j_max, k]]

        if np.isclose(Ab[k, k], 0):
            raise ValueError("Pivô nulo encontrado — sistema singular.")

        for i in range(k + 1, n):
            fator = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= fator * Ab[k, k:]

        if mostrar_passos:
            print(f"Após eliminação da coluna {k + 1}:")
            print(Ab, "\n")

    # --- Substituição regressiva ---
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - np.dot(Ab[i, i + 1:], x[i + 1:])) / Ab[i, i]

    print("\n✅ Solução obtida por Eliminação de Gauss:")
    return x


# MÉTODOS ALTERNATIVOS

def resolver_por_svd(A, b):
    """Usa SVD para resolver sistemas singulares ou mal-condicionados"""
    from numpy.linalg import svd
    U, S, Vt = svd(A)
    S_inv = np.array([1/s if s > 1e-12 else 0 for s in S])
    A_pinv = Vt.T @ np.diag(S_inv) @ U.T
    x = A_pinv @ b
    print("\n✅ Solução obtida por SVD (não é Eliminação de Gauss):")
    return x.flatten()


def resolver_por_minimos_quadrados(A, b):
    """Usa método dos mínimos quadrados (equações normais)"""
    AtA = A.T @ A
    Atb = A.T @ b
    x = np.linalg.solve(AtA, Atb)
    print("\n✅ Solução obtida por Mínimos Quadrados (não é Eliminação de Gauss):")
    return x.flatten()
