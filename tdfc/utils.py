import sys

def getPeriodMax(start_date, end_date, price):
    max = -float(sys.maxsize)
    for i in range(start_date, end_date):
        if price[i] > max:
            max = price[i]
    return max

def getPeriodMin(start_date, end_date, price):
    min = float(sys.maxsize)
    for i in range(start_date, end_date):
        if price[i] < min:
            min = price[i]
    return min

def getPeriodMaxGrowth(start_date, end_date, price):
    max_growth = -float(sys.maxsize)
    begin = start_date
    end = start_date
    net = 0.0
    for i in range(start_date, end_date-1):
        end = i+1
        diff = price[i+1] - price[i]
        net += diff
        if net <= 0.0:
            net = 0.0
            begin = i
            continue
        if net > max_growth:
            max_growth = net
            max_begin = begin
            max_end = end
    return (max_begin+1, max_end)

def getPeriodMaxDrop(start_date, end_date, price):
    max_drop = float(sys.maxsize)
    begin = start_date
    end = start_date
    net = 0.0
    for i in range(start_date, end_date-1):
        end = i+1
        diff = price[i+1] - price[i]
        net += diff
        if net >= 0.0:
            net = 0.0
            begin = i
            continue
        if net < max_drop:
            max_drop = net
            max_begin = begin
            max_end = end
    return (max_begin, max_end)