import pandas as pd

def load_feature_importance(model):
    model.eval()
    lst = []
    df = pd.DataFrame()
    for name, param in model.named_parameters():
        if len(param.detach()) == 1 and "presingle" in name:
            weight = float(param.detach().cpu())
            if weight < 0:
                lst.append(0)
            else:
                lst.append(weight)
    S = sum(lst)
    for t in range(len(lst)):
        lst[t] = lst[t]/S
    df["index"] = [1, 2, 3]
    df["weight"] = lst
    print(df)
    df.to_excel(r"weight.xlsx",index=False)


