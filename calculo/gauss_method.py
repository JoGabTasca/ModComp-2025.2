import numpy as np
from numpy.linalg import cond, matrix_rank, svd, solve

def gauss_somente_web(A, b, cond_limite=1e5):
    
    try:
        # Garante que são arrays numpy para cálculos
        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float).reshape(-1, 1)
        m, n = A_np.shape
    except Exception as e:
        return {
            'status': 'erro_input',
            'solucao': None,
            'mensagem': f'Erro ao formatar os dados de entrada: {e}. Verifique se a matriz e o vetor estão corretos.'
        }

    # CASO 1: MATRIZ QUADRADA
    if m == n:
        if matrix_rank(A_np) < n:
            return {
                'status': 'singular',
                'solucao': None,
                'mensagem': 'Este sistema é SINGULAR (det(A)=0). Não pode ser resolvido por Eliminação de Gauss. Tente o método SVD.'
            }
        
        c = cond(A_np)
        mensagem_cond = f"Sistema bem condicionado (cond(A) = {c:.2f})."
        if c > cond_limite:
            mensagem_cond = f"Sistema mal condicionado (cond(A) = {c:.2e}). O resultado pode conter erros numéricos."

        try:
            x = _eliminacao_gauss_pura(A_np, b_np)
            return {
                'status': 'sucesso_gauss',
                'solucao': list(x), # Converte para lista para JSON/template
                'mensagem': f"Solução obtida por Eliminação de Gauss.\n{mensagem_cond}"
            }
        except ValueError as e:
            return {
                'status': 'erro',
                'solucao': None,
                'mensagem': f'Erro durante a eliminação: {e}'
            }

    # CASO 2: MATRIZ NÃO QUADRADA
    else:
        return {
            'status': 'nao_quadrado',
            'solucao': None,
            'mensagem': 'Este sistema NÃO é quadrado (m ≠ n). Não pode ser resolvido por Eliminação de Gauss. Tente Mínimos Quadrados.'
        }

def _eliminacao_gauss_pura(A, b):
    
    n = len(b)
    Ab = np.hstack([A, b]) 

    for k in range(n - 1):
        i_max = np.argmax(np.abs(Ab[k:n, k])) + k
        Ab[[k, i_max]] = Ab[[i_max, k]]
        
        if np.isclose(Ab[k, k], 0):
            raise ValueError("Pivô nulo encontrado — sistema singular.")

        for i in range(k + 1, n):
            fator = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= fator * Ab[k, k:] 

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = np.dot(Ab[i, i + 1:n], x[i + 1:n])
        x[i] = (Ab[i, -1] - soma) / Ab[i, i]

    return x.flatten()


def resolver_por_svd_web(A, b):
    """Versão Web do SVD. Retorna um dicionário de resultado."""
    try:
        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float).reshape(-1, 1)
        
        U, S, Vt = svd(A_np)
        S_inv = np.array([1/s if s > 1e-12 else 0 for s in S])
        S_inv_diag = np.zeros((A_np.shape[1], A_np.shape[0]))
        diag_len = min(A_np.shape[0], A_np.shape[1])
        S_inv_diag[:diag_len, :diag_len] = np.diag(S_inv)
        
        A_pinv = Vt.T @ S_inv_diag @ U.T
        x = A_pinv @ b_np
        
        return {
            'status': 'sucesso_svd',
            'solucao': list(x.flatten()),
            'mensagem': 'Solução obtida por SVD (Pseudo-inversa).'
        }
    except Exception as e:
        return {'status': 'erro', 'solucao': None, 'mensagem': f'Erro no SVD: {e}'}

def resolver_por_minimos_quadrados_web(A, b):
    """Versão Web do Mínimos Quadrados. Retorna um dicionário."""
    try:
        A_np = np.array(A, dtype=float)
        b_np = np.array(b, dtype=float).reshape(-1, 1)
        
        AtA = A_np.T @ A_np
        Atb = A_np.T @ b_np
        
        if matrix_rank(AtA) < AtA.shape[0]:
            return resolver_por_svd_web(A, b)
        
        x = solve(AtA, Atb)
        
        return {
            'status': 'sucesso_mq',
            'solucao': list(x.flatten()),
            'mensagem': 'Solução obtida por Mínimos Quadrados (Equações Normais).'
        }
    except Exception as e:
        return {'status': 'erro', 'solucao': None, 'mensagem': f'Erro nos Mínimos Quadrados: {e}'}

