import numpy as np
import scipy 
from image_relations import *
import pickle as pk
        

def build_graph( camera: tuple) -> ( dict ):
    rgb = camera[0]
    dep = camera[1]
    k_rgb = camera[2]
    k_depth = camera[3]
    r_depth2rgb = camera[4]
    t_depth2rgb = camera[5]


    w = 0
    world = rgb[w]

    RT_world = {}
    best_unfound = {}
    
    
    stack = list(range(0, len(rgb)))

    stack_dup = stack.copy()

    for i in stack_dup:
        R, T, percentage = transformation2cameras( ( rgb[0], rgb[i], dep[0], dep[i], k_rgb,  k_depth, r_depth2rgb, t_depth2rgb ), 0, i )
        if(percentage > 0.5):
            RT_world[i] = {
                "parent": w, 
                "R": R,
                "T": T
            } #The reference image (0) is the parent
            stack.remove(i)
        else:
            best_unfound[i] = {
                "parent": w, 
                "R": R,
                "T": T,
                "percentage": percentage
            } #The reference image (0) is the parent

    distance = 1

    def compare_stack_values(x):
        if x not in best_unfound:
            return -1
        else:
            return best_unfound[x]["percentage"]


    while((len(stack) > 0) and (distance < len(rgb))):
        print("\n\n\n -------- \n\n\n")
        print(distance)
        print("\n\n\n -------- \n\n\n")
        stack.sort(key=compare_stack_values, reverse=True)
        print(stack)
        print("\n\n\n -------- \n\n\n")

        for i in stack:
            for n in [i- distance, i+distance]:
                if (n > 0) and (n < len(rgb)):
                    R, T, percentage = transformation2cameras( ( rgb[n], rgb[i], dep[n], dep[i], k_rgb,  k_depth, r_depth2rgb, t_depth2rgb ), n, i )
                    print(i, n, percentage)
                    print("\n\n\n -------- \n\n\n")
                    if (percentage > 0.5) and (n in RT_world):
                            # Insert in graph
                            RT_world[i] = {
                                "parent": n, 
                                "R": R,
                                "T": T
                            }
                            stack.remove(i)
                            if i in best_unfound:
                                del best_unfound[i]

                            more_to_check = True
                            while(more_to_check):
                                more_to_check = False
                                keys_to_iterate = list(best_unfound.keys()).copy()
                                for point in keys_to_iterate: # check if new found value is parent of any best_unfound
                                    if (best_unfound[point]["parent"] in RT_world) and best_unfound[point]["percentage"] > 0.5:
                                        RT_world[point] = { # This does not work. Please check
                                            "parent" : best_unfound[point]["parent"], 
                                            "R" : best_unfound[point]["R"],
                                            "T" : best_unfound[point]["T"]
                                        }
                                        stack.remove(point)
                                        if point in best_unfound:
                                            del best_unfound[point]
                                        more_to_check = True
                            break
                    else:
                        # If its better than best_unfound, update
                        if (i in best_unfound) and (n not in RT_world):
                            if percentage > best_unfound[i]["percentage"]:
                                best_unfound[i] = {
                                    "parent": n, 
                                    "R": R,
                                    "T": T,
                                    "percentage": percentage
                                }
        distance += 1

    # Clean all up
    if(len(best_unfound.keys()) > 0):
        clean_best_unfound = True
        while(clean_best_unfound):
            clean_best_unfound = False
            keys_to_iterate = list(best_unfound.keys()).copy()
            for point in keys_to_iterate:
                i = point
                while(i > 0):
                    if best_unfound[point]["parent"] in RT_world:
                        RT_world[point] = { # This does not work. Please check
                            "parent" : best_unfound[point]["parent"], 
                            "R" : best_unfound[point]["R"],
                            "T" : best_unfound[point]["T"]
                        }
                        stack.remove(point)
                        if point in best_unfound:
                            del best_unfound[point]
                        more_to_check = True
                        break
                    elif (point-i in RT_world) and (point-i > 0):
                        R, T, percentage = transformation2cameras( ( rgb[point-i], rgb[point], dep[point-i], dep[point], k_rgb,  k_depth, r_depth2rgb, t_depth2rgb ), point-i, point, True )
                        RT_world[point] = {
                            "parent" : point - i, 
                            "R" : R,
                            "T" : T
                        }
                        stack.remove(point)
                        if point in best_unfound:
                            del best_unfound[point]
                        clean_best_unfound = True
                        break
                    i = i-1


    print(RT_world)
    print("\n\n\n -------------- \n\n\n")
    print(best_unfound)

    with open( "rt_graph.p", "wb" ) as file:
        pk.dump( RT_world, file, protocol=pk.HIGHEST_PROTOCOL )

    with open( "best_unfound.p", "wb" ) as file:
        pk.dump( best_unfound, file, protocol=pk.HIGHEST_PROTOCOL )


