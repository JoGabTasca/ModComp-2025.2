# calculo/bissecao_method.py

def metodo_bissecao(func, a, b, erro=1e-7, i_max=100, prev_m=None, iter_count=0):
    """
    Encontra a raiz de uma função usando o método da Bisseção (versão recursiva).
    Retorna: (raiz, iteracoes, f(raiz), atingiu_max_iter, erro_calculado)
    """
    # Primeira chamada: verifica se f(a) e f(b) têm sinais opostos
    if prev_m is None and iter_count == 0:
        y_a_inicial = func(a)
        y_b_inicial = func(b)
        if y_a_inicial * y_b_inicial >= 0:
            return None, 0, None, False, None
    
    y_a = func(a)
    y_b = func(b)
    m = (a + b) / 2.0
    y_m = func(m)
    
    iter_count += 1
    
    # Calcula o erro relativo
    erro_calculado = abs((m - prev_m)/m) if prev_m is not None else None
    
    # Critério de parada: erro relativo
    if prev_m is not None and erro_calculado < erro:
        return m, iter_count, y_m, False, erro_calculado
    
    # Critério de parada: f(m) = 0
    if y_m == 0:
        return m, iter_count, y_m, False, 0.0
    
    # Critério de parada: iterações esgotadas
    if iter_count >= i_max:
        return m, iter_count, y_m, True, erro_calculado
    
    # Escolhe o novo intervalo
    if (y_a * y_m < 0):
        b = m
    elif (y_b * y_m < 0):
        a = m
    else:
        return m, iter_count, y_m, False, erro_calculado
    
    return metodo_bissecao(func, a, b, erro, i_max, m, iter_count)