import pickle

count = 0
b0 = pickle.load(open("224soldata/imagebatch0.pkl", 'rb'))
print("Loaded", count)
count+=1
b1 = pickle.load(open("224soldata/imagebatch1.pkl", 'rb'))
print("Loaded", count)
count+=1
b2 = pickle.load(open("224soldata/imagebatch2.pkl", 'rb'))
print("Loaded", count)
count+=1
b3 = pickle.load(open("224soldata/imagebatch3.pkl", 'rb'))
print("Loaded", count)
count+=1
b4 = pickle.load(open("224soldata/imagebatch4.pkl", 'rb'))
print("Loaded", count)
count+=1
b5 = pickle.load(open("224soldata/imagebatch5.pkl", 'rb'))
print("Loaded", count)
count+=1

pickle.dump(b0+b1+b2+b3+b4+b5, open("224traindata/images.pkl", 'wb'))