import requests
import time
import execjs
import argparse
import math
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import base64

import os

VERBOSE = False

def getUrl(fscode):
    head = b'aHR0cDovL2Z1bmQuZWFzdG1vbmV5LmNvbS9waW5nemhvbmdkYXRhLw=='
    head = str(base64.decodebytes(head), 'utf-8')
    tail = '.js?v='+ time.strftime("%Y%m%d%H%M%S",time.localtime())
    return head+fscode+tail

def getWorth(fscode):
    content = requests.get(getUrl(fscode))

    year = ''
    for i in range(30, 34):
        year += content.text[i]
    month = ''
    for i in range(35, 37):
        month += content.text[i]
    day = ''
    for i in range(38, 40):
        day += content.text[i]
    info_date = year + month + day
    sys_date = time.strftime("%Y%m%d",time.localtime())
    verify_date = info_date == sys_date

    jsContent = execjs.compile(content.text)
    name = jsContent.eval('fS_name')
    code = jsContent.eval('fS_code')

    netWorthTrend = jsContent.eval('Data_netWorthTrend')

    ACWorthTrend = jsContent.eval('Data_ACWorthTrend')

    netWorth = []
    ACWorth = []

    for dayWorth in netWorthTrend[::-1]:
        netWorth.append(dayWorth['y'])

    for dayACWorth in ACWorthTrend[::-1]:
        ACWorth.append(dayACWorth[1])
    print(name,code)
    return netWorth, ACWorth, verify_date, info_date

class DayNumber:
    def __init__(self, input_days, output_days):
        self.input_days = input_days
        self.output_days = output_days

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.linear1 = nn.Linear(input_size, 16)
        self.relu = nn.ReLU(inplace=True)
        self.linear1_1 = nn.Linear(16, 64)
        self.lstm = nn.LSTM(64, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        input_seq = self.linear1(input_seq.view(len(input_seq), -1))
        input_seq = self.relu(input_seq)
        input_seq = self.linear1_1(input_seq.view(len(input_seq), -1))
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear2(lstm_out.view(len(input_seq), -1))
        return predictions

def train_lstm(price, PATH, day_number):
    inputs = []
    targets = []
    price = [(price[i + 1] - price[i])/price[i] for i in range(0, len(price) - 1)]
    for i in range(len(price)-(day_number.input_days + day_number.output_days - 1)):
        inputs.append([price[i:i+day_number.input_days]])
        targets.append([price[i+day_number.input_days:i+day_number.input_days+day_number.output_days]])
 
    inputs = torch.tensor(inputs, dtype=torch.float).reshape([len(inputs), 1, day_number.input_days])
    targets = torch.tensor(targets, dtype=torch.float).reshape([len(inputs), 1, day_number.output_days])

    lstm = LSTM(input_size=day_number.input_days, hidden_layer_size=64, output_size=day_number.output_days)
    lstm.hidden_cell = (torch.zeros(1, 1, lstm.hidden_layer_size), torch.zeros(1, 1, lstm.hidden_layer_size))
    if os.path.exists("./models/lstm/model.pth"):
        lstm.load_state_dict(torch.load("./models/lstm/model.pth"))

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm.parameters(), lr=0.00001)

    epochs = 100
    for i in range(epochs):
        for input, target in zip(inputs, targets):
            input = input.reshape([1, 1, day_number.input_days])
            target = target.reshape([1, 1, day_number.output_days])
            optimizer.zero_grad()

            lstm.hidden_cell = (torch.zeros(1, 1, lstm.hidden_layer_size),
                             torch.zeros(1, 1, lstm.hidden_layer_size))

            y = lstm(input)
            
            single_loss = loss_function(y, target)
            single_loss.backward(retain_graph=True)
            optimizer.step()

            if i%25 == 1 and VERBOSE:
                print(y, target)

        # if i%25 == 1:
        if single_loss < 0.000001: #0.000'001
            break
        print(f'epoch:{i:3} loss:{single_loss.item():10.8f}')
    print(f'epoch:{i:3} loss:{single_loss.item():10.10f}')

    torch.save(lstm.state_dict(), PATH)

def predict_lstm(price, day_number):
    inputs = []
    targets = []
    price = [(price[i + 1] - price[i])/price[i] for i in range(0, len(price) - 1)]
    for i in range(len(price)-(day_number.input_days - 1)):
        inputs.append([price[i:i+day_number.input_days]])
    for i in range(day_number.input_days, len(price)):
        targets.append(price[i])
 
    inputs = torch.tensor(inputs, dtype=torch.float).reshape([len(inputs), 1, day_number.input_days])

    lstm = LSTM(input_size=day_number.input_days, hidden_layer_size=64, output_size=day_number.output_days)
    lstm.hidden_cell = (torch.zeros(1, 1, lstm.hidden_layer_size), torch.zeros(1, 1, lstm.hidden_layer_size))
    if os.path.exists("./models/lstm/model.pth"):
        lstm.load_state_dict(torch.load("./models/lstm/model.pth"))

    lstm_result = []
    for input in inputs:
        input = input.reshape([1, 1, day_number.input_days])

        lstm.hidden_cell = (torch.zeros(1, 1, lstm.hidden_layer_size),
                            torch.zeros(1, 1, lstm.hidden_layer_size))

        y = lstm(input)
        lstm_result.append(y.detach().numpy()[0][0])

    for i in range(1, day_number.output_days):
        lstm_result.append(y.detach().numpy()[0][i])

    assert len(lstm_result) == len(targets) + day_number.output_days

    targets_final = [1.0]
    lstm_final = [1.0]
    target_value = 1.0
    lstm_value = 1.0

    for i in range(len(targets)):
        target_value *= (1+targets[i])
        targets_final.append(target_value)

    for i in range(len(lstm_result)):
        lstm_value *= (1+lstm_result[i])
        lstm_final.append(lstm_value)

    targets_final_norm = []
    target_max = max(targets_final)
    target_min = min(targets_final)
    for i in targets_final:
        targets_final_norm.append((i-target_min)/(target_max-target_min))

    lstm_final_norm = []
    lstm_max = max(lstm_final)
    lstm_min = min(lstm_final)
    for i in lstm_final:
        lstm_final_norm.append((i-lstm_min)/(lstm_max-lstm_min))

    # x_target = [i for i in range(len(targets))]
    # x_lstm = [i for i in range(len(lstm_result))]
    # plt.plot(x_target, targets, marker='o', markersize=3)
    # plt.plot(x_lstm, lstm_result, marker='o', markersize=3)

    x_target = [i for i in range(len(targets_final))]
    x_lstm = [i for i in range(len(lstm_final))]
    plt.plot(x_target, targets_final_norm, marker='o', markersize=3)
    plt.plot(x_lstm, lstm_final_norm, marker='o', markersize=3)

    plt.legend(['price', 'lstm'])
    plt.show()

def calculate(start_date, end_date, price, verbose=True, plot=False, LIMIT=500, date_verified=False, info_date='xxxxxxxx'):
    print("VERBOSE MODE:", verbose)

    money = []
    stock = 0.00001
    paid = 0.00001
    amount = 0
    daily = 0.0

    for i in range(start_date, end_date):
        paid += daily
        amount += daily / price[i]

        avg_rate_7 = (price[i] - price[i - 6]) / price[i - 6]
        avg_rate_7 = abs(avg_rate_7)

        avg_rate_15 = (price[i] - price[i - 14]) / price[i - 14]
        avg_rate_15 = abs(avg_rate_15)
        
        avg_rate_45 = (price[i] - price[i - 44]) / price[i - 44]
        avg_rate_45 = abs(avg_rate_45)

        avg_rate_58 = (price[i] - price[i - 57]) / price[i - 57]
        avg_rate_58 = abs(avg_rate_58)

        avg_rate_76 = (price[i] - price[i - 75]) / price[i - 75]
        avg_rate_76 = abs(avg_rate_76)
        
        avg_rate_90 = (price[i] - price[i - 89]) / price[i - 89]
        avg_rate_90 = abs(avg_rate_90)

        slope_score = 15.0**(0.16 * avg_rate_7 + 0.13 * avg_rate_15 + 0.15 * avg_rate_45 + 0.25 * avg_rate_58 + 0.24 * avg_rate_76 + 0.07 * avg_rate_90) - 1
        slope_score = min(1.0, slope_score)

        daily = (1 - slope_score) * LIMIT
        money.append(daily)

        stock = amount * price[i]

    if date_verified == True:
        print("Today to pay ({date})".format(date=time.strftime("%Y%m%d",time.localtime())), daily)
    else:
        a = input("Outdated info date, show yesterdays daily pay? (y/N)")
        if a == 'y':
            print(str(daily), 'paid yesterday ({pre_date})'.format(pre_date=info_date))

    if verbose == True:
        print("Price", price[-1])
        print("Stock", stock)
        print("Paid", paid)
        print("Net", stock - paid, str((stock - paid) * 100 / paid) + '%')
    print("Rate change", "today" if date_verified else "yesterday", '(' + time.strftime("%Y%m%d",time.localtime()) + ')',
     (price[-1] - price[-2])/price[-2])
    print('\n')

    if plot == True:
        max_price = max(price[start_date : end_date])
        min_price = min(price[start_date : end_date])
        price_norm = [(i - min_price) / (max_price - min_price) for i in price[start_date : end_date]]
        money_norm = [i / LIMIT for i in money]
        date_range = [i for i in range(end_date - start_date)]

        plt.plot(date_range, price_norm, marker='o', markersize=3)
        plt.plot(date_range, money_norm, marker='o', markersize=3)
        plt.legend(['price', 'daily'])
        plt.show()

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--code", "-c", default='005939') # Enter the six digit code
    argparser.add_argument("--mode", "-m", default='predict')
    argparser.add_argument("--limit", "-l", default=500)
    argparser.add_argument("--verbose", "-v", default=False)
    args = argparser.parse_args()
    fscode = args.code
    limit = float(args.limit)

    netWorth, ACWorth, date_verified, info_date = getWorth(fscode)
    price = netWorth[::-1]
    
    price = price[:]
    # price = price[:len(price)-10]

    days = len(price)
    # assert days > (365 + 90)

    # d = DayNumber(29, 4)
    d = DayNumber(60, 10)

    if args.mode == 'predict':
        # calculate(days-21, days, price, bool(args.verbose), True, limit, date_verified, info_date) # start from 1 month ago
        calculate(days-63, days, price, bool(args.verbose), True, limit, date_verified, info_date) # start from 3 month ago
        # calculate(days-126, days, price, bool(args.verbose), True, limit, date_verified, info_date) # start from 6 months ago
        # calculate(days-21*12, days, price, bool(args.verbose), True, limit, date_verified, info_date) # start from 1 year ago
        show_lstm_pred = input("Show trend prediction? (y/N)")
        if show_lstm_pred == 'y':
            predict_lstm(price, d)
    if args.mode == 'train_lstm':
        path="./models/lstm/model.pth"
        train_lstm(price, path, d)
