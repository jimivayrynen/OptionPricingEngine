from .black_scholes import BlackScholes

def calculate_implied_volatility(market_price, S, K, T, r, option_type='call'):
    """
    Calculates Implied Volatility (IV) using the Newton-Raphson method.
    Iteratively finds the volatility that matches the market price.
    """
    max_iter = 100    # Maximum iterations
    precision = 1.0e-5 # Target precision
    
    # Initial guess for volatility
    sigma = 0.5
    
    for i in range(max_iter):
        # Create model with current guess
        bs = BlackScholes(S, K, T, r, sigma)
        
        # Calculate price and Vega
        price = bs.calculate_price(option_type)
        vega = bs.calculate_greeks(option_type)['vega']
        
        # Calculate error
        diff = market_price - price
        
        if abs(diff) < precision:
            return sigma
        
        if abs(vega) < 1e-8:
            return None
        
        # Newton-Raphson update step
        try:
            sigma = sigma + diff / vega
        except ZeroDivisionError:
            return None
            
    return sigma