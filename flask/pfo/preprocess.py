def find_all(data, string):
    result = []
    for i in range(len(data)):
        if data[i].find(string) != -1:
            result.append(i)
    
    return result

def get_value(string):
    if string.find('=') != -1:
        return string[string.find('=')+1:]

def get_pfo_info(data):
    pfo = dict()
    
    ticker = [get_value(data[i]) for i in find_all(data, "asset_ticker")]
    name = [get_value(data[i]) for i in find_all(data, "asset_name")]
    weight = [float(get_value(data[i])) for i in find_all(data, "asset_weight")]
    
    pfo['tickers'] = ticker
    pfo['names'] = name
    pfo['weights'] = weight
    
    return pfo

def save_pfos(data, idx):
    result = dict()
    for i in range(len(idx)):
        if i != len(idx)-1:
            temp = [data[j] for j in range(idx[i], idx[i+1])]
            result[get_value(temp[0])] = get_pfo_info(temp)
        else:
            temp = data[idx[i]:]
            result[get_value(temp[0])] = get_pfo_info(temp)
    
    return result