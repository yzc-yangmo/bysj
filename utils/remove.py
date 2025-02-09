import os

if __name__ == '__main__':
    for i in os.listdir("./"):
        if i.endswith(".jpg"):
            os.remove(i)
            print(f"{i} remove successfully!")