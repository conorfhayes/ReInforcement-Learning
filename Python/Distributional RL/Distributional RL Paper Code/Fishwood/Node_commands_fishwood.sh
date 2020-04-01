python3 Node_DistRL.py --name Node_DistRL --env FishWood-v0 --episodes 10000 --avg 10 --hidden 50 --lr 0.001 --ret forward --extra-state none --utility "min(r1, r2 // 2)" > /dev/null
