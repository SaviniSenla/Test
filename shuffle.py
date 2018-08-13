import random
with open('G:\Dialog\Data Set\BezdekIris.data.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('G:\Dialog\Data Set\shuffled.txt','w') as target:
    for _, line in data:
        target.write( line )
        
print("The Change")