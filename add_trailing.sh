#!/bin/bash
    FILE="engine/trade_engine.py"`1
# Add init (8 spaces indent)
awk '1;/self.is_running = True/{print "        self.trade_manager = TradeManager(self.broker)"; print; next}1' $FILE > tmp.py && mv tmp.py $FILE

# Add call (12 spaces indent) after the while line
awk '1;/while self.is_running:/{print; print "            self.trade_manager.manage_open_trades()  # trailing manager - every cycle"; next}1' $FILE > tmp.py && mv tmp.py $FILE
