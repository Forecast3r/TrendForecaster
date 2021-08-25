import sys
sys.path.append("..")
from tdfc import utils

def test_getPeriodMax():
    price = [-1.0, -2.0, -3.0, 2.0, 1.0]
    max_price = utils.getPeriodMax(0, len(price), price)
    assert max_price == 2.0, "[Fail] {0}".format(sys._getframe().f_code.co_name)
    print("[Pass] {0}".format(sys._getframe().f_code.co_name))

def test_getPeriodMin():
    price = [-1.0, -2.0, -3.0, 2.0, 1.0]
    min_price = utils.getPeriodMin(0, len(price), price)
    assert min_price == -3.0, "[Fail] {0}".format(sys._getframe().f_code.co_name)
    print("[Pass] {0}".format(sys._getframe().f_code.co_name))

def test_getPeriodMaxGrowth():
    price = [1.0, 2.0, 2.0, -1.0, -2.0, -3.0, 2.0, 3.0, 4.0, 2.0]
    start_date, end_date = utils.getPeriodMaxGrowth(0, len(price), price)
    assert start_date == 5 and end_date == 8, "[Fail] {0}".format(sys._getframe().f_code.co_name)
    print("[Pass] {0}".format(sys._getframe().f_code.co_name))

def test_getPeriodMaxDrop():
    price = [1.0, 2.0, 2.0, -1.0, -2.0, -3.0, 2.0, 3.0, 4.0, 2.0]
    start_date, end_date = utils.getPeriodMaxDrop(0, len(price), price)
    assert start_date == 1 and end_date == 5, "[Fail] {0}".format(sys._getframe().f_code.co_name)
    print("[Pass] {0}".format(sys._getframe().f_code.co_name))

if __name__ == "__main__":
    test_getPeriodMax()
    test_getPeriodMin()
    test_getPeriodMaxGrowth()
    test_getPeriodMaxDrop()