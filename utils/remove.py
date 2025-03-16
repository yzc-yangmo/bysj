import os

def remove_not_jpg(tragrt_path):
    count = 0
    for f in os.listdir(tragrt_path):
        if not f.endswith(".jpg"):
            os.remove(os.path.join(tragrt_path, f))
            count += 1
            print(f"count: {count} {f} remove!")
    if count == 0:
        print("All files's type is jpg!")

if __name__ == '__main__':
    target_path = input("Enter the target path: ")
    remove_not_jpg(target_path)