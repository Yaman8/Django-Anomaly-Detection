from multiprocessing import Pool
def fun(x):
    return "this was : "+str(x) 

if __name__=="__main__":
    print("hello there...")
    with Pool(4) as p:
        for i in p.imap(fun, [1,3,5,7,9]):                        
            print(i) 