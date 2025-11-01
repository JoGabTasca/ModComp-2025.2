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
            # Em um front-end, 'input' pode precisar ser substituído por um 
            # componente de UI que retorne 's' ou 'n'.
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
            try:
                # Chama a função de eliminação corrigida
                return eliminacao_gauss(A, b, mostrar_passos)
            except ValueError as e:
                print(f"\nErro durante a eliminação: {e}")
                print("Isso pode ocorrer em sistemas singulares que passaram na verificação de rank devido a erros de ponto flutuante.")
                return None


    # CASO 2️: MATRIZ NÃO QUADRADA
    else:
        print("\n⚠️ Este sistema NÃO é quadrado (m ≠ n).")
        print("Esse problema NÃO pode ser resolvido por Eliminação de Gauss.")
        print("Método sugerido: Mínimos Quadrados (Normal Equations).")
        # Em um front-end, 'input' pode precisar ser substituído
        opcao = input("Deseja continuar com o método dos Mínimos Quadrados? (s/n): ").strip().lower()
        if opcao == 's':
            return resolver_por_minimos_quadrados(A, b)
        else:
            print("Operação encerrada. Use um sistema quadrado para aplicar Eliminação de Gauss.")
            return None


def eliminacao_gauss(A, b, mostrar_passos=False):
    """
    🔹 Implementação da Eliminação de Gauss com PIVOTEAMENTO PARCIAL.
    (Usada apenas em sistemas quadrados e não singulares)
    
    Esta versão está CORRIGIDA.
    """
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    n = len(b)
    # Matriz aumentada [A|b]
    Ab = np.hstack([A, b]) 

    if mostrar_passos:
        print("\nMatriz aumentada inicial:")
        print(Ab, "\n")

    # --- Etapa de eliminação progressiva (com Pivoteamento Parcial) ---
    for k in range(n - 1):
        # 1. Encontra o índice da linha com o maior pivô na coluna k
        #    (Procurando de k até n na coluna k)
        i_max = np.argmax(np.abs(Ab[k:n, k])) + k

        # 2. Troca de linhas (sem troca de colunas)
        Ab[[k, i_max]] = Ab[[i_max, k]]
        
        # 3. Verifica se o pivô (após a troca) é nulo
        if np.isclose(Ab[k, k], 0):
            # Se isso acontecer, a matriz é singular
            raise ValueError("Pivô nulo encontrado — sistema singular.")

        # 4. Eliminação
        for i in range(k + 1, n):
            fator = Ab[i, k] / Ab[k, k]
            # Atualiza toda a linha i (da coluna k em diante)
            Ab[i, k:] -= fator * Ab[k, k:] 

        if mostrar_passos:
            print(f"Após eliminação da coluna {k + 1}:")
            print(Ab, "\n")

    # --- Substituição regressiva (CORRIGIDA) ---
    # Esta é a seção que corrigiu o seu ValueError
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        # A soma vai da coluna i+1 até n-1 (índice 'n')
        # Ab[i, i + 1:n] -> Pega os elementos de A na linha i
        # x[i + 1:n]     -> Pega os elementos de x já calculados
        #
        # O slice Ab[i, i + 1:n] fica vazio quando i = n-1, 
        # e np.dot([], []) retorna 0.0, que é o correto.
        soma = np.dot(Ab[i, i + 1:n], x[i + 1:n])
        
        # Ab[i, -1] é o elemento n+1 (o b_i modificado)
        x[i] = (Ab[i, -1] - soma) / Ab[i, i]

    print("\n✅ Solução obtida por Eliminação de Gauss:")
    return x


# MÉTODOS ALTERNATIVOS (Estavam corretos, mantidos como estão)

def resolver_por_svd(A, b):
    """Usa SVD para resolver sistemas singulares ou mal-condicionados"""
    from numpy.linalg import svd
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    
    U, S, Vt = svd(A)
    
    # Cria a matriz S_inv (1/s) com tolerância para valores muito pequenos
    S_inv = np.array([1/s if s > 1e-12 else 0 for s in S])
    
    # Recria a matriz diagonal Sigma inversa (forma m x n)
    # A matriz Sigma original é m x n. Sua pseudo-inversa é n x m.
    S_inv_diag = np.zeros((A.shape[1], A.shape[0]))
    
    # Preenche a diagonal principal da pseudo-inversa
    diag_len = min(A.shape[0], A.shape[1])
    S_inv_diag[:diag_len, :diag_len] = np.diag(S_inv)
    
    # A_pinv = V @ S_inv_diag @ U.T
    A_pinv = Vt.T @ S_inv_diag @ U.T
    
    x = A_pinv @ b
    print("\n✅ Solução obtida por SVD (não é Eliminação de Gauss):")
    return x.flatten()


def resolver_por_minimos_quadrados(A, b):
    """Usa método dos mínimos quadrados (equações normais)"""
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float).reshape(-1, 1)
    
    # Equações Normais: (A.T @ A) @ x = (A.T @ b)
    AtA = A.T @ A
    Atb = A.T @ b
    
    # Verifica o condicionamento de A.T @ A antes de resolver
    if np.linalg.matrix_rank(AtA) < AtA.shape[0]:
        print("⚠️ Matriz A.T @ A é singular. Usando SVD para Mínimos Quadrados.")
        # Se AtA é singular, usamos SVD na matriz A original
        return resolver_por_svd(A, b)
    
    x = np.linalg.solve(AtA, Atb)
    print("\n✅ Solução obtida por Mínimos Quadrados (não é Eliminação de Gauss):")
    return x.flatten()
