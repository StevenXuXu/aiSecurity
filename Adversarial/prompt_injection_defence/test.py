import time

start = time.time()

sum = 0
for i in range(100000000):
    sum = i

end = time.time()

print(str(end - start))