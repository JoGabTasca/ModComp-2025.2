# calculo/newton_method.py

def newton_raphson(func, func_derivada, x, erro=1e-7, i_max=100, prev_x=None, iter_count=0):
    """
    Encontra a raiz de uma função usando o método de Newton-Raphson (versão recursiva).
    Retorna: (raiz, iteracoes, f(raiz), atingiu_max_iter, erro_calculado)
    """
    f_x = func(x)
    df_x = func_derivada(x)
    
    iter_count += 1
    
    # Verifica divisão por zero
    if df_x == 0:
        return None, iter_count, None, False, None
    
    # Calcula o erro relativo
    erro_calculado = abs((x - prev_x)/x) if prev_x is not None else None
    
    # Critério de parada: erro relativo
    if prev_x is not None and erro_calculado < erro:
        return x, iter_count, f_x, False, erro_calculado
    
    # Critério de parada: f(x) = 0
    if f_x == 0:
        return x, iter_count, f_x, False, 0.0
    
    # Critério de parada: iterações esgotadas
    if iter_count >= i_max:
        return x, iter_count, f_x, True, erro_calculado
    
    # Calcula o próximo x
    x_new = x - f_x / df_x
    
    return newton_raphson(func, func_derivada, x_new, erro, i_max, x, iter_count)