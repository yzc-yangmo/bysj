import os, threading

temp = {
    
  "apple_pie": 0,
  "baby_back_ribs": 1,
  "baklava": 2,
  "beef_carpaccio": 3,
  "beef_tartare": 4,
  "beet_salad": 5,
  "beignets": 6,
  "bibimbap": 7,
  "bread_pudding": 8,
  "breakfast_burrito": 9,
  "bruschetta": 10,
  "caesar_salad": 11,
  "cannoli": 12,
  "caprese_salad": 13,
  "carrot_cake": 14,
  "ceviche": 15,
  "cheesecake": 16,
  "cheese_plate": 17,
  "chicken_curry": 18,
  "chicken_quesadilla": 19,
  "chicken_wings": 20,
  "chocolate_cake": 21,
  "chocolate_mousse": 22,
  "churros": 23,
  "clam_chowder": 24,
  "club_sandwich": 25,
  "crab_cakes": 26,
  "creme_brulee": 27,
  "croque_madame": 28,
  "cup_cakes": 29,
  "deviled_eggs": 30,
  "donuts": 31,
  "dumplings": 32,
  "edamame": 33,
  "eggs_benedict": 34,
  "escargots": 35,
  "falafel": 36,
  "filet_mignon": 37,
  "fish_and_chips": 38,
  "foie_gras": 39,
  "french_fries": 40,
  "french_onion_soup": 41,
  "french_toast": 42,
  "fried_calamari": 43,
  "fried_rice": 44,
  "frozen_yogurt": 45,
  "garlic_bread": 46,
  "gnocchi": 47,
  "greek_salad": 48,
  "grilled_cheese_sandwich": 49,
  "grilled_salmon": 50,
  "guacamole": 51,
  "gyoza": 52,
  "hamburger": 53,
  "hot_and_sour_soup": 54,
  "hot_dog": 55,
  "huevos_rancheros": 56,
  "hummus": 57,
  "ice_cream": 58,
  "lasagna": 59,
  "lobster_bisque": 60,
  "lobster_roll_sandwich": 61,
  "macaroni_and_cheese": 62,
  "macarons": 63,
  "miso_soup": 64,
  "mussels": 65,
  "nachos": 66,
  "omelette": 67,
  "onion_rings": 68,
  "oysters": 69,
  "pad_thai": 70,
  "paella": 71,
  "pancakes": 72,
  "panna_cotta": 73,
  "peking_duck": 74,
  "pho": 75,
  "pizza": 76,
  "pork_chop": 77,
  "poutine": 78,
  "prime_rib": 79,
  "pulled_pork_sandwich": 80,
  "ramen": 81,
  "ravioli": 82,
  "red_velvet_cake": 83,
  "risotto": 84,
  "samosa": 85,
  "sashimi": 86,
  "scallops": 87,
  "seaweed_salad": 88,
  "shrimp_and_grits": 89,
  "spaghetti_bolognese": 90,
  "spaghetti_carbonara": 91,
  "spring_rolls": 92,
  "steak": 93,
  "strawberry_shortcake": 94,
  "sushi": 95,
  "tacos": 96,
  "takoyaki": 97,
  "tiramisu": 98,
  "tuna_tartare": 99,
  "waffles": 100
}

mapping = {v: k for k, v in temp.items()}


def work(rename_path, file_names, thread_id):
    try:
        for index, file in enumerate(file_names):
            if file.endswith(".jpg"):
                old_file_name = file
                info = file.split("-")
                if not info[1].isdigit():
                        continue
                new_file_name = "-".join([info[0], mapping[int(info[1])], info[2]])
                os.rename(os.path.join(rename_path, old_file_name), os.path.join(rename_path, new_file_name))
                print(f"thread_id: {thread_id} {(index+1)/len(file_names)*100:.2f}%   {[index+1]} / {len(file_names)},  {old_file_name} -> {new_file_name}" )
    except Exception as e:
        print(f"Error: {e}")
        return


def main(rename_path):
    current_jpgs = [i for i in os.listdir(rename_path) if i.endswith("jpg")]
    threads = []
    thread_num = 20
    step = len(current_jpgs) // thread_num
    
    for index, i in enumerate(range(0, len(current_jpgs), step)):
        t = threading.Thread(target=work, args=(rename_path, current_jpgs[i:i+step], index+1))
        threads.append(t)
        t.start()
      
    # 等待所有线程完成
    for t in threads:
        t.join()
    
    print("All processing completed!")
    
    
if __name__ == '__main__':
    main(r"./dataset/train")
    main(r"./dataset/val")