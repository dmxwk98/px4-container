import math
import numpy as np
class collision_check:
    def collision_check(Map, from_wp, to_wp) :

        N_grid = len(Map) + 5000

        min_x = math.floor(min(np.round(from_wp[0]), np.round(to_wp[0])))
        max_x = math.ceil(max(np.round(from_wp[0]), np.round(to_wp[0])))
        min_y = math.floor(min(np.round(from_wp[1]), np.round(to_wp[1])))
        max_y = math.ceil(max(np.round(from_wp[1]), np.round(to_wp[1])))

        if max_x > N_grid :
            max_x = N_grid
        if max_y > N_grid :
            max_y = N_grid

        check1 = Map[min_y][min_x]
        check2 = Map[min_y][max_x]
        check3 = Map[max_y][min_x]
        check4 = Map[max_y][max_x]

        flag_collision = max(check1, check2, check3, check4)

        return flag_collision
