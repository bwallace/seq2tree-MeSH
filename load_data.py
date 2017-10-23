import pandas as pd
import pickle 

def get_data(fname="data/pubmed.med.pkl"):
    with open("data/pubmed.med.pkl", 'rb') as input_f:
        df = pd.DataFrame(pickle.load(input_f))
        # need to map into unicode; they are bytes 
        # currently. @TODO should really fix this
        # on the output side.
        df["abstract"] = df["abstract"].str.decode("utf-8")
        return df

def assemble_pairs():
    ''' create input/output tensors '''
    df = get_data()
    inputs = df["abstract"].values
    targets = []
    # for now, simplify to just one target sequence
    for sequences in df["sequence"].values:
        targets.append(sequences[0][1])

    return (inputs, targets)



