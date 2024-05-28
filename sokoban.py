
import heapq
import random
import sys
import pygame
import queue
import copy
import time
import threading
from Game import Game
from Game2 import Game2
from pygame import mixer
mixer.init()

TIME_LIMITED = 1800
current_level = 1
node_generated= 0
start =0
end=0
solution=None
moves=0
depth_count=0
flag_countdown=True
sound_file_path = "song\start_song.mp3"
sound_enabled = True
ai2_play_done = False
ai1_play_done = False
stop_thread=True
class AlgorithmThread(threading.Thread):
    def __init__(self, game, algorithm):
        threading.Thread.__init__(self)
        self.game = game
        self.algorithm = algorithm
        self.result = None
        self.flag_auto = 1

    def run(self):
        if self.algorithm == "BFS Algorithm":
            self.result = BFSsolution(self.game)
        elif self.algorithm == "DFS Algorithm":
            self.result = DFSsolution(self.game)
        elif self.algorithm == "UCS Algorithm":
            self.result = UCSsolution(self.game)
        elif self.algorithm == "Greedy Algorithm":
            self.result = greedySolution(self.game)
        elif self.algorithm == "A* Algorithm":
            self.result = AstarSolution(self.game)
        elif self.algorithm == "ID Algorithm":
            self.result = IDSolution(self.game)
        elif self.algorithm == "Dijkstra Algorithm":
            self.result = DijkstraSolution(self.game)
        self.flag_auto = 0
def is_game_equal(game1, game2):
    return game1.get_matrix() == game2.get_matrix()
def AI1_play(game,algorithm_player1,screen):
    global ai1_play_done
    i = 0
    bool = True
    thread_player = AlgorithmThread(game, algorithm_player1)
    thread_player.start()
    while bool:
        while thread_player.is_alive() or bool:
            if not thread_player.is_alive() and i < len(thread_player.result):
                playByBotPlayer1(game, thread_player.result[i], screen)
                i += 1
                if i == len(thread_player.result):
                    display_player_win(screen, "AI")
                    if sound_enabled:
                        sound_file_path = './song/start_song.mp3'
                        play_sound(sound_file_path)
                    bool = False
    ai1_play_done=True
def run_AI2_play(game2, algorithm_player2, onevsonewindown):
    AI2_play(game2, algorithm_player2, onevsonewindown)
def run_AI1_play(game, algorithm_player1, onevsonewindown):
    AI1_play(game, algorithm_player1, onevsonewindown)    
def AI2_play(game2,algorithm_player2,screen):
    global ai2_play_done,stop_thread
    i = 0
    bool = True
    thread_player = AlgorithmThread(game2, algorithm_player2)
    thread_player.start()
    while bool :
        while thread_player.is_alive() or bool :
            if stop_thread==False:
                break
            if not thread_player.is_alive() and i < len(thread_player.result):
                playByBotPlayer2(game2, thread_player.result[i], screen)
                i += 1
                if i == len(thread_player.result):
                    display_player_win(screen, "AI")
                    if sound_enabled:
                        sound_file_path = './song/start_song.mp3'
                        play_sound(sound_file_path)
                    bool = False
    ai2_play_done = True
def solo_game(game, algorithm_player1, game2, algorithm_player2, screen):
    i = 0
    j = 0
    bool = True
    bool = True
    flag2 = True
    thread_player1 = AlgorithmThread(game2, algorithm_player2)
    thread_player2 = AlgorithmThread(game, algorithm_player1)

    thread_player1.start()
    thread_player2.start()

    while bool:
        while thread_player1.is_alive() or thread_player2.is_alive() or bool:
            if not thread_player1.is_alive() and i < len(thread_player1.result):
                playByBotPlayer1(game, thread_player1.result[i], screen)
                i += 1
                if i == len(thread_player1.result) and j != len(thread_player2.result):
                    
                    if sound_enabled:
                        sound_file_path = './song/start_song.mp3'
                        play_sound(sound_file_path)
                    flag1=False
                    bool = False
            if not thread_player2.is_alive() and j < len(thread_player2.result):
                playByBotPlayer2(game2, thread_player2.result[j], screen)
                j += 1
                if j == len(thread_player2.result) and i != len(thread_player1.result):
                    
                    if sound_enabled:
                        sound_file_path = './song/start_song.mp3'
                        play_sound(sound_file_path)
                    flag2=False
                    bool = False

        # Kiểm tra trạng thái hòa
        if i == len(thread_player1.result) and j == len(thread_player2.result):
            if is_game_equal(game, game2):  # Thêm hàm is_game_equal để kiểm tra trạng thái hòa
                display_player_win(screen, "draw")
                bool = False
        elif flag1==False:
            display_player_win(screen, "player1")
        elif flag2==False:
            display_player_win(screen, "player2")


def nextLevelonevsone():
    global current_level
    if current_level <30:
        current_level += 1
        initsoloLevel()
def backLevelonevsone():
    global current_level
    if current_level > 1:
        current_level -= 1
        initsoloLevel()
def nextLevel():
        global current_level
        if current_level<30:
            current_level += 1
        initLevel()
def backLevel():
    global current_level
    if current_level > 1:
        current_level -= 1
        initLevel()
def undo_move(game,screen):
    game.unmove()
    print_game(game.get_matrix(), screen)
def undo_move1(game,screen):
    game.unmove()
    print_game_a_b(game.get_matrix(), screen,0,0)
def undo_move_player(game2,screen):
    game2.unmove()
    print_game_a_b_player2(game2.get_matrix(), screen,600,0)
def validMove(state):
    x = 0
    y = 0
    move = []
    for step in ["U","D","L","R"]:
        if step == "U":
            x = 0
            y = -1
        elif step == "D":
            x = 0
            y = 1
        elif step == "L":
            x = -1
            y = 0
        elif step == "R":
            x = 1
            y = 0

        if state.can_move(x,y) or state.can_push(x,y):
            move.append(step)

    return move

''' Check deadlock: box on the corner of walls or other boxes   '''
def is_deadlock(state):
    box_list = state.box_list()
    for box in box_list:
        x = box[0]
        y = box[1]
        #corner up-left
        if state.get_content(x,y-1) in ['#','$','*'] and state.get_content(x-1,y) in ['#','$','*']:
            if state.get_content(x-1,y-1) in ['#','$','*']:
                return True
            if state.get_content(x,y-1) == '#' and state.get_content(x-1,y) =='#':
                return True
            if state.get_content(x,y-1) in ['$','*'] and state.get_content(x-1,y) in ['$','*']:
                if state.get_content(x+1,y-1) == '#' and state.get_content(x-1,y+1) == '#':
                    return True
            if state.get_content(x,y-1) in ['$','*'] and state.get_content(x-1,y) == '#':
                if state.get_content(x+1,y-1) == '#':
                    return True
            if state.get_content(x,y-1) == '#' and state.get_content(x-1,y) in ['$','*']:
                if state.get_content(x-1,y+1) == '#':
                    return True
                
        #corner up-right
        if state.get_content(x,y-1) in ['#','$','*'] and state.get_content(x+1,y) in ['#','$','*']:
            if state.get_content(x+1,y-1) in ['#','$','*']:
                return True
            if state.get_content(x,y-1) == '#' and state.get_content(x+1,y) =='#':
                return True
            if state.get_content(x,y-1) in ['$','*'] and state.get_content(x+1,y) in ['$','*']:
                if state.get_content(x-1,y-1) == '#' and state.get_content(x+1,y+1) == '#':
                    return True
            if state.get_content(x,y-1) in ['$','*'] and state.get_content(x+1,y) == '#':
                if state.get_content(x-1,y-1) == '#':
                    return True
            if state.get_content(x,y-1) == '#' and state.get_content(x+1,y) in ['$','*']:
                if state.get_content(x+1,y+1) == '#':
                    return True


        #corner down-left
        elif state.get_content(x,y+1) in ['#','$','*'] and state.get_content(x-1,y) in ['#','$','*']:
            if state.get_content(x-1,y+1) in ['#','$','*']:
                return True
            if state.get_content(x,y+1) == '#' and state.get_content(x-1,y) =='#':
                return True
            if state.get_content(x,y+1) in ['$','*'] and state.get_content(x-1,y) in ['$','*']:
                if state.get_content(x-1,y-1) == '#' and state.get_content(x+1,y+1) == '#':
                    return True
            if state.get_content(x,y+1) in ['$','*'] and state.get_content(x-1,y) == '#':
                if state.get_content(x+1,y+1) == '#':
                    return True
            if state.get_content(x,y+1) == '#' and state.get_content(x-1,y) in ['$','*']:
                if state.get_content(x-1,y-1) == '#':
                    return True
                

        #corner down-right
        elif state.get_content(x,y+1) in ['#','$','*'] and state.get_content(x+1,y) in ['#','$','*']:
            if state.get_content(x+1,y+1) in ['#','$','*']:
                return True
            if state.get_content(x,y+1) == '#' and state.get_content(x+1,y) =='#':
                return True
            if state.get_content(x,y+1) in ['$','*'] and state.get_content(x+1,y) in ['$','*']:
                if state.get_content(x-1,y+1) == '#' and state.get_content(x+1,y-1) == '#':
                    return True
            if state.get_content(x,y+1) in ['$','*'] and state.get_content(x+1,y) == '#':
                if state.get_content(x-1,y+1) == '#':
                    return True
            if state.get_content(x,y+1) == '#' and state.get_content(x+1,y) in ['$','*']:
                if state.get_content(x+1,y-1) == '#':
                    return True
                
    return False


def get_distance(state, a_star=True):
    sum = 0
    box_list = state.box_list()
    dock_list = state.dock_list()

    # Check if box_list and dock_list are not empty
    if box_list and dock_list:
        for box in box_list:
            for dock in dock_list:
                sum += (abs(dock[0] - box[0]) + abs(dock[1] - box[1]))

        if not a_star:
            # Nếu không phải A*, giảm giá trị để ảnh hưởng ít hơn đến   
            sum = sum // 2

    return sum

def heuristic_a_star(state):
    # Tính toán chi phí từ worker đến các box
    worker = state.worker()
    box_list = state.box_list()
    dock_list = state.dock_list()

    total_cost = 0

    for box in box_list:
        if dock_list:  # Kiểm tra nếu dock_list không rỗng
            # Chi phí từ box đến dock gần nhất
            cost_box_to_dock = min([abs(box[0] - dock[0]) + abs(box[1] - dock[1]) for dock in dock_list])
            total_cost += cost_box_to_dock
        else:
            # Trong trường hợp không có docks, chỉ tính chi phí từ worker đến box
            total_cost += abs(worker[0] - box[0]) + abs(worker[1] - box[1])

    return total_cost

def heuristic_greedy(state):
    worker = state.worker()
    box_list = state.box_list()
    
    total_cost = 0

    for box in box_list:
        # Tính toán khoảng cách Euclidean giữa worker và box
        cost_worker_to_box = ((worker[0] - box[0]) ** 2 + (worker[1] - box[1]) ** 2) ** 0.5
        total_cost += cost_worker_to_box

    return total_cost


def worker_to_box(state, a_star=True):
    p = 1000
    worker = state.worker()
    box_list = state.box_list()
    for box in box_list:
        if (abs(worker[0] - box[0]) + abs(worker[1] - box[1])) <= p:
            p = abs(worker[0] - box[0]) + abs(worker[1] - box[1])
    if not a_star:
        p = p // 2
    return p

def play_sound(file_path):
    mixer.music.load(file_path)
    mixer.music.play()

  
def stop_sound():
    pygame.mixer.music.stop()
def clear_speaker(screen):
    background_color = screen.get_at((20, 20))  
    pygame.draw.rect(screen, background_color, (20, 20, 50, 50))
def run_BFS_threaded(game,screen):
    global node_generated, start, end, solution, moves
    font_path = './font/fontpixel.ttf'
    font2 = pygame.font.Font(font_path, 13)
    node_generated = 0
    start = 0
    end = 0
    solution = None
    sol = BFSsolution(game)
    text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
    text_rect2 = text2.get_rect(topleft=(120, 120))
    text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
    text_rect3 = text3.get_rect(topleft=(120, 90))
    text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
    text_rect4 = text4.get_rect(topleft=(120, 60))
    text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
    text_rect5 = text5.get_rect(topleft=(120, 150))
    screen.blit(text5, text_rect5)
    screen.blit(text2, text_rect2)
    screen.blit(text3, text_rect3)
    screen.blit(text4, text_rect4)

    flagAuto = 1
    i = 0
    while flagAuto and (i < len(sol)):
        playByBot(game, sol[i], screen)
        i += 1
        if i == len(sol):
            flagAuto = 0
        time.sleep(0.1)
def run_DFS_threaded(game,screen):
    global node_generated, start, end, solution, moves
    font_path = './font/fontpixel.ttf'
    font2 = pygame.font.Font(font_path, 13)
    node_generated = 0
    start = 0
    end = 0
    solution = None
    sol = DFSsolution(game)
    text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
    text_rect2 = text2.get_rect(topleft=(120, 120))
    text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
    text_rect3 = text3.get_rect(topleft=(120, 90))
    text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
    text_rect4 = text4.get_rect(topleft=(120, 60))
    text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
    text_rect5 = text5.get_rect(topleft=(120, 150))
    screen.blit(text5, text_rect5)
    screen.blit(text2, text_rect2)
    screen.blit(text3, text_rect3)
    screen.blit(text4, text_rect4)

    flagAuto = 1
    i = 0
    while flagAuto and (i < len(sol)):
        playByBot(game, sol[i], screen)
        i += 1
        if i == len(sol):
            flagAuto = 0
        time.sleep(0.1)
def run_UCS_threaded(game,screen):
    global node_generated, start, end, solution, moves
    font_path = './font/fontpixel.ttf'
    font2 = pygame.font.Font(font_path, 13)
    node_generated = 0
    start = 0
    end = 0
    solution = None
    sol = UCSsolution(game)
    text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
    text_rect2 = text2.get_rect(topleft=(120, 120))
    text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
    text_rect3 = text3.get_rect(topleft=(120, 90))
    text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
    text_rect4 = text4.get_rect(topleft=(120, 60))
    text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
    text_rect5 = text5.get_rect(topleft=(120, 150))
    screen.blit(text5, text_rect5)
    screen.blit(text2, text_rect2)
    screen.blit(text3, text_rect3)
    screen.blit(text4, text_rect4)

    flagAuto = 1
    i = 0
    while flagAuto and (i < len(sol)):
        playByBot(game, sol[i], screen)
        i += 1
        if i == len(sol):
            flagAuto = 0
        time.sleep(0.1)
def run_greedy_threaded(game,screen):
    global node_generated, start, end, solution, moves
    font_path = './font/fontpixel.ttf'
    font2 = pygame.font.Font(font_path, 13)
    node_generated = 0
    start = 0
    end = 0
    solution = None
    sol = greedySolution(game)
    text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
    text_rect2 = text2.get_rect(topleft=(120, 120))
    text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
    text_rect3 = text3.get_rect(topleft=(120, 90))
    text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
    text_rect4 = text4.get_rect(topleft=(120, 60))
    text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
    text_rect5 = text5.get_rect(topleft=(120, 150))
    screen.blit(text5, text_rect5)
    screen.blit(text2, text_rect2)
    screen.blit(text3, text_rect3)
    screen.blit(text4, text_rect4)

    flagAuto = 1
    i = 0
    while flagAuto and (i < len(sol)):
        playByBot(game, sol[i], screen)
        i += 1
        if i == len(sol):
            flagAuto = 0
        time.sleep(0.1)
def run_astar_threaded(game,screen):
    global node_generated, start, end, solution, moves
    font_path = './font/fontpixel.ttf'
    font2 = pygame.font.Font(font_path, 13)
    node_generated = 0
    start = 0
    end = 0
    solution = None
    sol = AstarSolution(game)
    text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
    text_rect2 = text2.get_rect(topleft=(120, 120))
    text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
    text_rect3 = text3.get_rect(topleft=(120, 90))
    text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
    text_rect4 = text4.get_rect(topleft=(120, 60))
    text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
    text_rect5 = text5.get_rect(topleft=(120, 150))
    screen.blit(text5, text_rect5)
    screen.blit(text2, text_rect2)
    screen.blit(text3, text_rect3)
    screen.blit(text4, text_rect4)

    flagAuto = 1
    i = 0
    while flagAuto and (i < len(sol)):
        playByBot(game, sol[i], screen)
        i += 1
        if i == len(sol):
            flagAuto = 0
        time.sleep(0.1)
def run_dijsktra_threaded(game,screen):
    global node_generated, start, end, solution, moves
    font_path = './font/fontpixel.ttf'
    font2 = pygame.font.Font(font_path, 13)
    node_generated = 0
    start = 0
    end = 0
    solution = None
    sol = DijkstraSolution(game)
    text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
    text_rect2 = text2.get_rect(topleft=(120, 120))
    text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
    text_rect3 = text3.get_rect(topleft=(120, 90))
    text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
    text_rect4 = text4.get_rect(topleft=(120, 60))
    text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
    text_rect5 = text5.get_rect(topleft=(120, 150))
    screen.blit(text5, text_rect5)
    screen.blit(text2, text_rect2)
    screen.blit(text3, text_rect3)
    screen.blit(text4, text_rect4)

    flagAuto = 1
    i = 0
    while flagAuto and (i < len(sol)):
        playByBot(game, sol[i], screen)
        i += 1
        if i == len(sol):
            flagAuto = 0
        time.sleep(0.1)
def run_BFS_threaded(game,screen):
    global node_generated, start, end, solution, moves
    font_path = './font/fontpixel.ttf'
    font2 = pygame.font.Font(font_path, 13)
    node_generated = 0
    start = 0
    end = 0
    solution = None
    sol = BFSsolution(game)
    text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
    text_rect2 = text2.get_rect(topleft=(120, 120))
    text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
    text_rect3 = text3.get_rect(topleft=(120, 90))
    text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
    text_rect4 = text4.get_rect(topleft=(120, 60))
    text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
    text_rect5 = text5.get_rect(topleft=(120, 150))
    screen.blit(text5, text_rect5)
    screen.blit(text2, text_rect2)
    screen.blit(text3, text_rect3)
    screen.blit(text4, text_rect4)

    flagAuto = 1
    i = 0
    while flagAuto and (i < len(sol)):
        playByBot(game, sol[i], screen)
        i += 1
        if i == len(sol):
            flagAuto = 0
        time.sleep(0.1)
def IDSolution(game):
    global node_generated, start, end, solution, moves,depth_count
    start = time.time()
    node_generated = 0
    moves = 0
    depth_count = 1

    while True:
        result = DLSsolution(game, depth_count)
        
        if result == "NoSol":
            # Nếu không tìm thấy giải pháp ở độ sâu hiện tại, tăng độ sâu và thử lại
            depth_count += 1
        elif result == "TimeOut":
            return "TimeOut"
        else:
            return result

def DLSsolution(game, depth_limit):
    global node_generated, start, end, solution, moves,depth_count
    start = time.time()
    node_generated = 0
    moves = 0
    state = copy.deepcopy(game)  # Parent Node
    node_generated += 1
    
    if is_deadlock(state):
        end = time.time()
        solution = "No solution"
        return "NoSol"

    stateSet = [state]  # Stack to store traversed nodes
    stateExplored = []  # List of visited nodes (store matrix of nodes)
    
    
    
    while stateSet:
        if (time.time() - start) >= TIME_LIMITED:
            print("Time Out!")
            return "TimeOut"
        
        currState = stateSet.pop()  # Get the top node of the stack to be the current node
        move = validMove(currState)  # Find next valid moves of current node in type of list of char ["U","D","L","R"]
        stateExplored.append(currState.get_matrix())  # Add matrix of current node to visited list

        for step in move:
            newState = copy.deepcopy(currState)
            node_generated += 1
            
            if step == "U":
                newState.move(0, -1, False)
            elif step == "D":
                newState.move(0, 1, False)
            elif step == "L":
                newState.move(-1, 0, False)
            elif step == "R":
                newState.move(1, 0, False)

            newState.pathSol += step
            
            if newState.is_completed():
                end = time.time()
                solution = newState.pathSol
                moves = len(solution)
                return newState.pathSol

            if (newState.get_matrix() not in stateExplored) and (not is_deadlock(newState)) and (len(newState.pathSol) <= depth_limit):
                stateSet.append(newState)

    end = time.time()
    solution = "No solution"
    return "NoSol"

def BFSsolution(game):
    global node_generated,start,end,solution,moves
    start = time.time()
    state = copy.deepcopy(game) # Parent Node                 
    node_generated += 1
    moves=0
    if is_deadlock(state):
        end = time.time()
        solution="No Solution"
        return "NoSol"
    stateSet = queue.Queue()    # Queue to store traversed nodes 
    stateSet.put(state)
    stateExplored = []          # list of visited node (store matrix of nodes)
    print("Processing...")
    while not stateSet.empty():
        if (time.time() - start) >= TIME_LIMITED:
            print("Time Out!")
            return "TimeOut"                    
        currState = stateSet.get()                      # get the top node of the queue to be the current node
        move = validMove(currState)                     # find next valid moves of current node in type of list of char ["U","D","L","R"]
        stateExplored.append(currState.get_matrix())    # add matrix of current node to visited list
        random.shuffle(move)
        for step in move:                               
            newState = copy.deepcopy(currState)
            node_generated += 1
            if step == "U":
                newState.move(0,-1,False)
            elif step == "D":
                newState.move(0,1,False)
            elif step == "L":
                newState.move(-1,0,False)
            elif step == "R":
                newState.move(1,0,False)
            newState.pathSol += step
        
            if newState.is_completed():
                end = time.time()

                solution=newState.pathSol
                moves=len(solution)
                return newState.pathSol

            if (newState.get_matrix() not in stateExplored) and (not is_deadlock(newState)):
                stateSet.put(newState)
    end = time.time()
    solution="No solution"
    return "NoSol"
def heuristic_a_star(state):
    # Hàm heuristic cho A*
    worker = state.worker()
    box_list = state.box_list()
    dock_list = state.dock_list()

    # Tính khoảng cách lớn nhất từ worker đến hộp và từ hộp đến ô đích gần nhất
    max_worker_to_box = 0
    for box in box_list:
        distance = abs(worker[0] - box[0]) + abs(worker[1] - box[1])
        max_worker_to_box = max(max_worker_to_box, distance)

    min_box_to_dock = float('inf')
    for box in box_list:
        for dock in dock_list:
            distance = abs(box[0] - dock[0]) + abs(box[1] - dock[1])
            min_box_to_dock = min(min_box_to_dock, distance)

    return max_worker_to_box + min_box_to_dock

def get_cost(state, step):
    # Hàm tính chi phí cho mỗi bước di chuyển
    # Bạn có thể điều chỉnh chi phí ở đây nếu cần
    return 1

# Hàm A* tối ưu nhất

def AstarSolution(game):
    global node_generated, start, end, solution, moves
    start = time.time()
    node_generated = 0
    moves = 0
    state = copy.deepcopy(game)  # Node cha
    state.heuristic = heuristic_a_star(state)
    node_generated += 1

    if is_deadlock(state):
        end = time.time()
        solution = "No solution"
        return "NoSol"

    stateSet = queue.PriorityQueue()  # Hàng đợi ưu tiên để lưu trữ các nút đã duyệt (giá thấp -> ưu tiên cao)
    stateSet.put((state.heuristic, state))
    stateExplored = set()  # Danh sách các nút đã duyệt (lưu ma trận của các nút)
    stateExplored.add(tuple(map(tuple, state.get_matrix())))  # Chuyển ma trận thành tuple có thể băm được và thêm vào danh sách

    print("Processing...")

    while not stateSet.empty():
        if (time.time() - start) >= TIME_LIMITED:
            print("Time Out!")
            return "TimeOut"

        currCost, currState = stateSet.get()  # Lấy nút đầu tiên của hàng đợi để làm nút hiện tại
        move = validMove(currState)  # Tìm các bước di chuyển hợp lệ tiếp theo của nút hiện tại
        stateExplored.add(tuple(map(tuple, currState.get_matrix())))  # Thêm ma trận của nút hiện tại vào danh sách đã duyệt

        for step in move:
            newState = copy.deepcopy(currState)
            node_generated += 1
            if step == "U":
                newState.move(0, -1, False)
            elif step == "D":
                newState.move(0, 1, False)
            elif step == "L":
                newState.move(-1, 0, False)
            elif step == "R":
                newState.move(1, 0, False)

            newState.pathSol += step
            newState.heuristic = heuristic_a_star(newState)
            newCost = currCost + get_cost(currState, step)  # Chi phí tăng thêm 1 với mỗi bước di chuyển

            if newState.is_completed():
                end = time.time()
                solution = newState.pathSol
                moves = len(solution)
                return newState.pathSol

            hashed_matrix = tuple(map(tuple, newState.get_matrix()))
            if hashed_matrix not in stateExplored and not is_deadlock(newState):
                stateSet.put((newCost + newState.heuristic, newState))

    end = time.time()
    solution = "No solution"
    return "NoSol"
def DijkstraSolution(game):
    global node_generated, start, end, solution, moves
    start = time.time()
    node_generated = 0
    state = copy.deepcopy(game)  # Parent Node
    state.heuristic = worker_to_box(state, a_star=False) + get_distance(state, a_star=False)
    node_generated += 1

    if is_deadlock(state):
        end = time.time()
        solution = "No solution"
        return "NoSol"

    stateSet = queue.PriorityQueue()  # Queue to store traversed nodes (low cost -> high priority)
    stateSet.put((0, state))
    stateExplored = set()  # set of visited nodes (store matrix of nodes)
    stateExplored.add(tuple(map(tuple, state.get_matrix())))  # convert matrix to hashable tuple and add to set

    print("Processing...")

    while not stateSet.empty():
        if (time.time() - start) >= TIME_LIMITED:
            print("Time Out!")
            return "TimeOut"

        currCost, currState = stateSet.get()  # get the top node of the queue to be the current node
        move = validMove(currState)  # find next valid moves of the current node in the type of a list of chars ["U","D","L","R"]

        for step in move:
            newState = copy.deepcopy(currState)
            node_generated += 1

            if step == "U":
                newState.move(0, -1, False)
            elif step == "D":
                newState.move(0, 1, False)
            elif step == "L":
                newState.move(-1, 0, False)
            elif step == "R":
                newState.move(1, 0, False)

            newState.pathSol += step
            newState.heuristic = worker_to_box(newState, a_star=False) + get_distance(newState, a_star=False)
            newCost = currCost + 1  # Cost increases by 1 for each move

            if newState.is_completed():
                end = time.time()
                solution = newState.pathSol
                moves = len(solution)
                return newState.pathSol

            hashed_matrix = tuple(map(tuple, newState.get_matrix()))
            if hashed_matrix not in stateExplored and not is_deadlock(newState):
                stateSet.put((newCost, newState))
                stateExplored.add(hashed_matrix)

    end = time.time()
    solution = "No solution"
    return "NoSol"
def heuristic_hill_climbing(state):
    worker = state.worker()
    box_list = state.box_list()
    dock_list = state.dock_list()

    # Kiểm tra xem có hộp nào còn ở xa nhất từ worker không
    max_distance = 0
    for box in box_list:
        distance = abs(worker[0] - box[0]) + abs(worker[1] - box[1])
        max_distance = max(max_distance, distance)

    # Nếu có hộp còn xa, trả về khoảng cách đó, ngược lại trả về 0
    return max_distance
def DFSsolution(game):
    global node_generated,start,end,solution,moves
    start = time.time()
    node_generated = 0
    moves=0
    state = copy.deepcopy(game)  # Parent Node
    node_generated += 1
    if is_deadlock(state):
        end = time.time()
        solution="No solution"
        return "NoSol"
    stateSet = []  # Stack to store traversed nodes
    stateSet.append(state)
    stateExplored = []  # list of visited node (store matrix of nodes)
    print("Processing...")
    while stateSet:
        if (time.time() - start) >= TIME_LIMITED:
            print("Time Out!")
            return "TimeOut"
        currState = stateSet.pop()  # get the top node of the stack to be the current node
        move = validMove(currState)  # find next valid moves of current node in type of list of char ["U","D","L","R"]
        stateExplored.append(currState.get_matrix())
        random.shuffle(move)
        for step in move:
            newState = copy.deepcopy(currState)
            node_generated += 1
            if step == "U":
                newState.move(0, -1, False)
            elif step == "D":
                newState.move(0, 1, False)
            elif step == "L":
                newState.move(-1, 0, False)
            elif step == "R":
                newState.move(1, 0, False)
            newState.pathSol += step

            if newState.is_completed():
                end = time.time()
                solution = newState.pathSol
                moves=len(solution)
                return newState.pathSol

            if (newState.get_matrix() not in stateExplored) and (not is_deadlock(newState)):
                stateSet.append(newState)
    end = time.time()
    solution="No solution"
    return "NoSol"

def UCSsolution(game):
    global node_generated, start, end, solution, moves
    start = time.time()
    node_generated = 0
    moves = 0
    state = copy.deepcopy(game)  # Parent Node
    node_generated += 1
    if is_deadlock(state):
        end = time.time()
        solution = "No solution"
        return "NoSol"
    
    stateSet = queue.PriorityQueue()  # Queue to store traversed nodes (low cost -> high priority)
    stateSet.put((0, state))
    stateExplored = set()  # set of visited nodes (store matrix of nodes)
    stateExplored.add(tuple(map(tuple, state.get_matrix())))  # convert matrix to hashable tuple and add to set
    
    print("Processing...")

    while not stateSet.empty():
        if (time.time() - start) >= TIME_LIMITED:
            print("Time Out!")
            return "TimeOut"

        currCost, currState = stateSet.get()  # get the top node of the queue to be the current node
        move = validMove(currState)  # find next valid moves of current node in type of list of char ["U","D","L","R"]
        
        for step in move:
            newState = copy.deepcopy(currState)
            node_generated += 1

            if step == "U":
                newState.move(0, -1, False)
            elif step == "D":
                newState.move(0, 1, False)
            elif step == "L":
                newState.move(-1, 0, False)
            elif step == "R":
                newState.move(1, 0, False)

            newState.pathSol += step
            newCost = currCost + 1  # Chi phí tăng lên 1 với mỗi bước di chuyển

            if newState.is_completed():
                end = time.time()
                solution = newState.pathSol
                moves = len(solution)
                return newState.pathSol

            hashed_matrix = tuple(map(tuple, newState.get_matrix()))
            if hashed_matrix not in stateExplored and not is_deadlock(newState):
                stateSet.put((newCost, newState))
                stateExplored.add(hashed_matrix)

    end = time.time()
    solution = "No solution"
    return "NoSol"

def greedySolution(game):
    global node_generated, start, end, solution, moves
    start = time.time()
    node_generated = 0
    moves = 0
    state = copy.deepcopy(game)  # Parent Node
    state.heuristic = worker_to_box(state,a_star=False)+get_distance(state,a_star=False)
    node_generated += 1
    if is_deadlock(state):
        end = time.time()
        solution = "No solution"
        return "NoSol"
    stateSet = queue.PriorityQueue()  # Queue to store traversed nodes (low index -> high priority)
    stateSet.put(state)
    stateExplored = []  # list of visited node (store matrix of nodes)
    print("Processing...")
    while not stateSet.empty():
        if (time.time() - start) >= TIME_LIMITED:
            print("Time Out!")
            return "TimeOut"
        currState = stateSet.get()  # get the top node of the queue to be the current node
        move = validMove(currState)  # find next valid moves of current node in type of list of char ["U","D","L","R"]
        stateExplored.append(currState.get_matrix())  # add matrix of current node to visited list
        for step in move:
            newState = copy.deepcopy(currState)
            node_generated += 1
            if step == "U":
                newState.move(0, -1, False)
            elif step == "D":
                newState.move(0, 1, False)
            elif step == "L":
                newState.move(-1, 0, False)
            elif step == "R":
                newState.move(1, 0, False)
            newState.pathSol += step
            newState.heuristic = worker_to_box(newState,a_star=False)+get_distance(newState,a_star=False)

            if newState.is_completed():
                end = time.time()
                solution = newState.pathSol
                moves = len(solution)
                return newState.pathSol

            if (newState.get_matrix() not in stateExplored) and (not is_deadlock(newState)):
                stateSet.put(newState)
    end = time.time()
    solution = "No solution"
    return "NoSol"
import queue
import copy
import time
def start_game():
    start = pygame.display.set_mode((700,700))
    level = ask(start,"Select Level")
    if int (level) > 0:
        return level
    else:
        print("ERROR: Invalid Level: "+str(level))
        sys.exit(2)
def createLevelsWindow():
    global current_level
    levels_window = pygame.display.set_mode((840, 620))
    pygame.display.set_caption("Levels")
    buttons = []
    button_font = pygame.font.Font(None, 36)

    button_width = 800 // 6
    button_height = 600 // 5

    rows = 5
    cols = 6

    horizontal_spacing = 5
    vertical_spacing = 5

    current_level = 1
    for row in range(rows):
        for col in range(cols):
            if current_level <= 30:
                x = (button_width + horizontal_spacing) * col + horizontal_spacing
                y = (button_height + vertical_spacing) * row + vertical_spacing

                button = pygame.draw.rect(levels_window, (230, 230, 250), (x, y, button_width, button_height))

                button_text = button_font.render(str(current_level), True, (0, 0, 0))
                button_text_rect = button_text.get_rect(center=button.center)
                levels_window.blit(button_text, button_text_rect.topleft)

                buttons.append((button, current_level))  # Lưu button và level tương ứng

                current_level += 1

    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Kiểm tra xem có phải là nút trái chuột được nhấn không
                    mouse_x, mouse_y = event.pos
                    for button, level in buttons:
                        if button.collidepoint(mouse_x, mouse_y):
                            play_level(level)
def createSoloLevelsWindow():
    global current_level
    levels_window = pygame.display.set_mode((840, 620))
    pygame.display.set_caption("Levels")
    buttons = []
    button_font = pygame.font.Font(None, 36)

    button_width = 800 // 6
    button_height = 600 // 5

    rows = 5
    cols = 6

    horizontal_spacing = 5
    vertical_spacing = 5

    current_level = 1
    for row in range(rows):
        for col in range(cols):
            if current_level <= 30:
                x = (button_width + horizontal_spacing) * col + horizontal_spacing
                y = (button_height + vertical_spacing) * row + vertical_spacing

                button = pygame.draw.rect(levels_window, (64, 224, 208), (x, y, button_width, button_height))

                button_text = button_font.render(str(current_level), True, (0, 0, 0))
                button_text_rect = button_text.get_rect(center=button.center)
                levels_window.blit(button_text, button_text_rect.topleft)

                buttons.append((button, current_level))  # Lưu button và level tương ứng

                current_level += 1

    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Kiểm tra xem có phải là nút trái chuột được nhấn không
                    mouse_x, mouse_y = event.pos
                    for button, level in buttons:
                        if button.collidepoint(mouse_x, mouse_y):
                            playsolo_level(level)

def draw_speaker(screen):
    if sound_enabled==True:
        speaker_image = pygame.image.load("images/audio_on.png") 
    else:
        speaker_image = pygame.image.load("images/audio_off.png")
    speaker_rect = speaker_image.get_rect()
    speaker_rect.topleft = (20, 20)
    screen.blit(speaker_image, speaker_rect)
def play_level(level):
    global current_level
    current_level = level
    initLevel()
def playsolo_level(level):
    global current_level
    current_level = level
    initsoloLevel()
def createOneVsOneWindow():
    initsoloLevel()
def createWelcomWindow():
    welcome_window = pygame.display.set_mode((800, 600))
    global continue_button
    background_image = pygame.image.load("./images/background_welcome.png")
    background_image = pygame.transform.scale(background_image, (800, 600))
    welcome_window.blit(background_image, (0, 0))
    font = pygame.font.Font("font/font.ttf", 50)
    font2 = pygame.font.Font("font/font.ttf", 30)
    play_sound(sound_file_path)
    text = font.render(f"HCMUTE", True, (199, 21 ,133))
    text_rect = text.get_rect(center=(400, 100))
    text1 = font2.render(f"Lưu Thế Quyền Anh", True, (255 ,204, 153))
    text_rect1 = text1.get_rect(center=(400, 200))
    text2 = font2.render(f"2110124", True,  (255 ,204, 153))
    text_rect2 = text2.get_rect(center=(400, 300))
    text3 = font2.render(f"Project AI", True, (255 ,204, 153))
    text_rect3 = text3.get_rect(center=(400, 400))
    border_rect = pygame.Rect(text_rect.left - 10, text_rect.top - 10, text_rect.width + 20, text_rect.height + 20)
    border_rect1 = pygame.Rect(text_rect1.left - 10, text_rect1.top - 10, text_rect1.width + 20, text_rect1.height + 20)
    border_rect2 = pygame.Rect(text_rect2.left - 10, text_rect2.top - 10, text_rect2.width + 20, text_rect2.height + 20)
    border_rect3 = pygame.Rect(text_rect3.left - 10, text_rect3.top - 10, text_rect3.width + 20, text_rect3.height + 20)
    pygame.draw.rect(welcome_window, (255,228,196), border_rect, 2)
    pygame.draw.rect(welcome_window, (255,228,196), border_rect1, 2)
    pygame.draw.rect(welcome_window, (255,228,196), border_rect2, 2)
    pygame.draw.rect(welcome_window, (255,228,196), border_rect3, 2)
    
    continue_button = pygame.draw.rect(welcome_window, (139, 71, 93), (250, 500, 300, 50))
    continue_text = font.render("Continue", True, (0, 0, 0))
    continue_text_rect = continue_text.get_rect(center=continue_button.center)
    welcome_window.blit(text, text_rect)
    welcome_window.blit(text1, text_rect1)
    welcome_window.blit(text2, text_rect2)
    welcome_window.blit(text3, text_rect3)
    welcome_window.blit(continue_text, continue_text_rect.topleft)
    pygame.display.set_caption("Project Sokoban")
    pygame.display.update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if continue_button.collidepoint(event.pos):
                    createMenuWindow()
def createMenuWindow():
    menu_window = pygame.display.set_mode((800, 600))
    global play_button, levels_button, solo_button,quit_button,sound_enabled
    background_image = pygame.image.load("./images/menu_background.png")
    background_image = pygame.transform.scale(background_image, (800, 600))
    menu_window.blit(background_image, (0, 0))
    font = pygame.font.Font(None, 36)
    draw_speaker(menu_window)
    image = pygame.image.load("images/play_button.png")
    image = pygame.transform.scale(image, (220, 82))
    image1 = pygame.image.load("images/level_button.png")
    image1 = pygame.transform.scale(image1, (220, 82))
    image2 = pygame.image.load("images/solo_button.png")
    image2 = pygame.transform.scale(image2, (220, 82))
    image3 = pygame.image.load("images/quit_button.png")
    image3 = pygame.transform.scale(image3, (220, 82))
    play_button = pygame.draw.rect(menu_window, (0, 0, 0,0), (300, 170, 200, 50))
    levels_button = pygame.draw.rect(menu_window, (0,0,0,0), (300, 280, 200, 50))
    solo_button = pygame.draw.rect(menu_window, (0,0,0,0), (300, 390, 200, 50))
    quit_button = pygame.draw.rect(menu_window, (0,0,0,0), (300, 500, 200, 50))
    menu_window.blit(image, (290, 160))
    menu_window.blit(image1, (290, 270))
    menu_window.blit(image2, (290, 380))
    menu_window.blit(image3, (290, 490))
    pygame.display.set_caption("Menu")
    pygame.display.update()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONDOWN:
                if play_button.collidepoint(event.pos):
                    createPlayModeWindow()
                elif levels_button.collidepoint(event.pos):
                    createLevelsWindow()
                elif solo_button.collidepoint(event.pos):
                    createOneVsOneWindow()
                elif 20 <= event.pos[0] <= 90 and 20 <= event.pos[1] <= 70:
                    sound_enabled = not sound_enabled
                    if sound_enabled==True:
                        play_sound(sound_file_path)
                        clear_speaker(menu_window)
                        draw_speaker(menu_window)
                        pygame.display.update()
                    elif sound_enabled==False:
                        stop_sound()
                        clear_speaker(menu_window)
                        draw_speaker(menu_window)
                        pygame.display.update()

                        
                elif quit_button.collidepoint(event.pos):
                    pygame.display.quit()
                    sys.exit()

    return menu_window
def is_counting_down():
    return pygame.event.peek(pygame.QUIT) or pygame.event.peek(pygame.KEYDOWN)
def countdown_screen(onevsonewindown):
    global flag_countdown
    font_path = './font/fontpixel.ttf'
    font = pygame.font.Font(font_path, 100)
    countdown_texts = ["3", "2", "1"]

    # Lưu background trước khi bắt đầu đếm ngược
    background = onevsonewindown.copy()

    for text in countdown_texts:
        text_surface = font.render(text, True, (255, 255, 255))
        text_rect = text_surface.get_rect(center=(500, 300))

        onevsonewindown.blit(background, (0, 0))  # Sử dụng background đã lưu để khôi phục trạng thái ban đầu
        onevsonewindown.blit(text_surface, text_rect)
        pygame.display.flip()  # Cập nhật màn hình
        pygame.time.delay(1000)  # Delay 1 giây

        pygame.time.delay(500)  # Delay 0.5 giây để tạo khoảng trắng giữa các số
    onevsonewindown.blit(background, (0, 0))
def initsoloLevel():
    global current_level,sound_file_path,sound_enabled,flag_countdown,stop_thread
    level=current_level
    game = Game(map_open('.\levels',level))
    game2 = Game2(map_open('.\levels',level))
    a=600
    b=0
    algorithms = ["Player 2","BFS Algorithm", "DFS Algorithm", "UCS Algorithm", "Greedy Algorithm", "A* Algorithm", "Dijkstra Algorithm", "ID Algorithm"]
    current_algorithm_index = 0
    algorithms2 = ["Player 1","BFS Algorithm", "DFS Algorithm", "UCS Algorithm", "Greedy Algorithm", "A* Algorithm", "Dijkstra Algorithm", "ID Algorithm"]
    current_algorithm_index2 = 0
    font_path = './font/fontpixel.ttf'
    font = pygame.font.Font(font_path, 15)
    onevsonewindown = pygame.display.set_mode((1000, 600))
    pygame.display.set_caption("Play Mode")
    background_image = pygame.image.load("images/onevsone.png")
    background_image = pygame.transform.scale(background_image, (1000, 600))
    onevsonewindown.blit(background_image, (0, 0))
    print_game_a_b(game.get_matrix(),onevsonewindown,0,0)
    print_game_a_b_player2(game2.get_matrix(),onevsonewindown,a,b)
    text_box_rect = pygame.Rect(20, 500, 300, 30)
    pygame.draw.rect(onevsonewindown, (230, 230, 250), text_box_rect)
    text_surface = font.render(algorithms[current_algorithm_index], True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=(text_box_rect.centerx, text_box_rect.centery))
    onevsonewindown.blit(text_surface, text_rect.topleft)
    
    text_box_rect2 = pygame.Rect(680, 500, 300, 30)
    pygame.draw.rect(onevsonewindown, (230, 230, 250), text_box_rect2)
    text_surface2 = font.render(algorithms2[current_algorithm_index2], True, (0, 0, 0))
    text_rect2 = text_surface.get_rect(center=(text_box_rect2.centerx, text_box_rect2.centery))
    onevsonewindown.blit(text_surface2, text_rect2.topleft)

    image = pygame.image.load("images/menu.png")
    image = pygame.transform.scale(image, (130, 55))
    image2 = pygame.image.load("images/fight.png")
    image2= pygame.transform.scale(image2, (135, 65))
    image3 = pygame.image.load("images/back.png")
    image3= pygame.transform.scale(image3, (120, 70))
    image4 = pygame.image.load("images/next.png")
    image4= pygame.transform.scale(image4, (120, 75))
    image5 = pygame.image.load("images/rematch.png")
    image5= pygame.transform.scale(image5, (150, 63))
    image6 = pygame.image.load("images/level.png")
    image6= pygame.transform.scale(image6, (140, 61))
    
    battle_button = pygame.draw.rect(onevsonewindown, (64, 224, 208), (420, 440, 100, 30))
    nextlevelonevsone_button = pygame.draw.rect(onevsonewindown, (64, 224, 208), (540, 440, 90, 30))
    backlevelonevsone_button = pygame.draw.rect(onevsonewindown, (64, 224, 208), (300, 445, 90, 30))
    backmenuonevsone_button = pygame.draw.rect(onevsonewindown, (64, 224, 208), (420, 390, 90, 30))
    resetplayer_button = pygame.draw.rect(onevsonewindown, (64, 224, 208), (335, 490, 110, 30))
    chooselevel_button = pygame.draw.rect(onevsonewindown,(64, 224, 208), (505,495,90,30))
    
    onevsonewindown.blit(image, (410, 380))
    onevsonewindown.blit(image2, (400, 425))    
    onevsonewindown.blit(image3, (290, 425))
    onevsonewindown.blit(image4, (530, 425))
    onevsonewindown.blit(image5, (325, 480))
    onevsonewindown.blit(image6, (490, 480))

    font_path = './font/fontpixel.ttf'
    font = pygame.font.Font(font_path, 15)
    text = font.render(f"Level: {current_level}", True, (255, 255, 255))
    text_rect = text.get_rect(center=(480, 50))
    onevsonewindown.blit(text, text_rect)
    algorithm_player1="Player 2"
    algorithm_player2 ="Player 1"
    flag1=True
    flag2=True
    while True:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w and flag_countdown==False: 
                    game.move(0,-1, True)
                    flag1=False
                    if game.is_completed():
                        display_player_win(onevsonewindown, "player2")
                        if sound_enabled:
                            sound_file_path = './song/start_song.mp3'
                            play_sound(sound_file_path)
                elif event.key == pygame.K_s and flag_countdown==False: 
                    game.move(0,1, True)
                    flag1=False
                    if game.is_completed():
                        display_player_win(onevsonewindown, "player2")
                        if sound_enabled:
                            sound_file_path = './song/start_song.mp3'
                            play_sound(sound_file_path)
                elif event.key == pygame.K_a and flag_countdown==False: 
                    game.move(-1,0, True)
                    flag1=False
                    if game.is_completed():
                        display_player_win(onevsonewindown, "player2")
                        if sound_enabled:
                            sound_file_path = './song/start_song.mp3'
                            play_sound(sound_file_path)
                elif event.key == pygame.K_d and flag_countdown==False: 
                    game.move(1,0, True)
                    flag1=False
                    if game.is_completed():
                       display_player_win(onevsonewindown, "player2")
                       if sound_enabled:
                            sound_file_path = './song/start_song.mp3'
                            play_sound(sound_file_path)
                elif event.key == pygame.K_LEFT and flag_countdown==False:
                    game2.move(-1,0, True)
                    flag2=False
                    if game2.is_completed():
                        display_player_win(onevsonewindown, "player1")
                        if sound_enabled:
                            sound_file_path = './song/start_song.mp3'
                            play_sound(sound_file_path)
                elif event.key == pygame.K_RIGHT and flag_countdown==False:
                    game2.move(1,0, True)
                    flag2=False
                    if game2.is_completed():
                        display_player_win(onevsonewindown, "player1")
                        if sound_enabled:
                            sound_file_path = './song/start_song.mp3'
                            play_sound(sound_file_path)
                elif event.key == pygame.K_DOWN and flag_countdown==False:
                    game2.move(0,1, True)
                    flag2=False
                    if game2.is_completed():
                        display_player_win(onevsonewindown, "player1")
                        if sound_enabled:
                            sound_file_path = './song/start_song.mp3'
                            play_sound(sound_file_path)
                elif event.key == pygame.K_UP and flag_countdown==False:
                    game2.move(0,-1, True)
                    flag2=False
                    if game2.is_completed():
                        display_player_win(onevsonewindown, "player1")
                        if sound_enabled:
                            sound_file_path = './song/start_song.mp3'
                            play_sound(sound_file_path)
                elif event.key == pygame.K_q:
                    undo_move1(game,onevsonewindown)
                elif event.key == pygame.K_m:
                    undo_move_player(game2,onevsonewindown)
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
            elif event.type ==pygame.MOUSEBUTTONDOWN:
                if backmenuonevsone_button.collidepoint(event.pos):
                    createMenuWindow()
                elif nextlevelonevsone_button.collidepoint(event.pos):
                    nextLevelonevsone()
                elif backlevelonevsone_button.collidepoint(event.pos):
                    backLevelonevsone()
                elif resetplayer_button.collidepoint(event.pos):
                    initsoloLevel()
                elif chooselevel_button.collidepoint(event.pos):
                    createSoloLevelsWindow()
                elif text_box_rect.collidepoint(event.pos):
                    current_algorithm_index = (current_algorithm_index + 1) % len(algorithms)
                    if current_algorithm_index > 8:
                        current_algorithm_index = 0
                    algorithm_player1=algorithms[current_algorithm_index]
                    text_box_rect = pygame.Rect(20, 500, 300, 30)
                    pygame.draw.rect(onevsonewindown, (230, 230, 250), text_box_rect)
                    text_surface = font.render(algorithms[current_algorithm_index], True, (0, 0, 0))
                    text_rect = text_surface.get_rect(center=(text_box_rect.centerx, text_box_rect.centery))
                    onevsonewindown.blit(text_surface, text_rect.topleft)
                elif text_box_rect2.collidepoint(event.pos):
                    current_algorithm_index2 = (current_algorithm_index2 + 1) % len(algorithms2)
                    if current_algorithm_index2 > 8:
                        current_algorithm_index2 = 0
                    algorithm_player2=algorithms2[current_algorithm_index2]
                    text_box_rect2 = pygame.Rect(680, 500, 300, 30)
                    pygame.draw.rect(onevsonewindown, (230, 230, 250), text_box_rect2)
                    text_surface2 = font.render(algorithms2[current_algorithm_index2], True, (0, 0, 0))
                    text_rect2 = text_surface2.get_rect(center=(text_box_rect2.centerx, text_box_rect2.centery))
                    onevsonewindown.blit(text_surface2, text_rect2.topleft)
                elif battle_button.collidepoint(event.pos):
                    if sound_enabled==True:
                        sound_file_path = './song/battle_song.mp3'
                        play_sound(sound_file_path)
                    if algorithm_player1!="Player 2" and algorithm_player2!="Player 1":
                        solo_game(game,algorithm_player1,game2,algorithm_player2,onevsonewindown)
                    elif algorithm_player1=="Player 2" and algorithm_player2=="Player 1":
                        countdown_screen(onevsonewindown)
                        flag_countdown=False
                    elif algorithm_player1!="Player 2" and algorithm_player2=="Player 1":
                        ai2_play_done = False
                        stop_thread = True
                        ai2_thread = threading.Thread(target=run_AI2_play, args=(game, algorithm_player1, onevsonewindown))
                        ai2_thread.start()
                        while not ai2_play_done:
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_LEFT :
                                        game2.move(-1,0, True)
                                        flag2=False
                                        if game2.is_completed():
                                            stop_thread=False
                                            display_player_win(onevsonewindown, "player1")
                                            if sound_enabled:
                                                sound_file_path = './song/start_song.mp3'
                                                play_sound(sound_file_path)
                                            break
                                    elif event.key == pygame.K_RIGHT :
                                        game2.move(1,0, True)
                                        flag2=False
                                        if game2.is_completed():
                                            stop_thread=False
                                            display_player_win(onevsonewindown, "player1")
                                            if sound_enabled:
                                                sound_file_path = './song/start_song.mp3'
                                                play_sound(sound_file_path)
                                    elif event.key == pygame.K_DOWN :
                                        game2.move(0,1, True)
                                        flag2=False
                                        if game2.is_completed():
                                            stop_thread=False
                                            display_player_win(onevsonewindown, "player1")
                                            if sound_enabled:
                                                sound_file_path = './song/start_song.mp3'
                                                play_sound(sound_file_path)
                                    elif event.key == pygame.K_UP :
                                        game2.move(0,-1, True)
                                        flag2=False
                                        if game2.is_completed():
                                            stop_thread=False
                                            display_player_win(onevsonewindown, "player1")
                                            if sound_enabled:
                                                sound_file_path = './song/start_song.mp3'
                                                play_sound(sound_file_path)
                                elif event.type ==pygame.MOUSEBUTTONDOWN:
                                    if backmenuonevsone_button.collidepoint(event.pos):
                                        createMenuWindow()
                                    elif nextlevelonevsone_button.collidepoint(event.pos):
                                        nextLevelonevsone()
                                    elif backlevelonevsone_button.collidepoint(event.pos):
                                        backLevelonevsone()
                                    elif resetplayer_button.collidepoint(event.pos):
                                        initsoloLevel()
                                    elif chooselevel_button.collidepoint(event.pos):
                                        createSoloLevelsWindow()
                            print_game_a_b_player2_no_update(game2.get_matrix(),onevsonewindown,a,b)
                            pygame.display.update()
                            
                    elif algorithm_player1=="Player 2" and algorithm_player2!="Player 1":
                        ai1_play_done = False
                        stop_thread = True
                        ai1_thread = threading.Thread(target=run_AI1_play, args=(game2, algorithm_player2, onevsonewindown))
                        ai1_thread.start()
                        while not ai1_play_done:
                            for event in pygame.event.get():
                                if event.type == pygame.KEYDOWN:
                                    if event.key == pygame.K_w : 
                                        game.move(0,-1, True)
                                        flag1=False
                                        if game.is_completed():
                                            stop_thread=False
                                            display_player_win(onevsonewindown, "player2")
                                            if sound_enabled:
                                                sound_file_path = './song/start_song.mp3'
                                                play_sound(sound_file_path)
                                    elif event.key == pygame.K_s : 
                                        game.move(0,1, True)
                                        flag1=False
                                        if game.is_completed():
                                            stop_thread=False
                                            display_player_win(onevsonewindown, "player2")
                                            if sound_enabled:
                                                sound_file_path = './song/start_song.mp3'
                                                play_sound(sound_file_path)
                                    elif event.key == pygame.K_a : 
                                        game.move(-1,0, True)
                                        flag1=False
                                        if game.is_completed():
                                            stop_thread=False
                                            display_player_win(onevsonewindown, "player2")
                                            if sound_enabled:
                                                sound_file_path = './song/start_song.mp3'
                                                play_sound(sound_file_path)
                                    elif event.key == pygame.K_d : 
                                        game.move(1,0, True)
                                        flag1=False
                                        if game.is_completed():
                                            stop_thread=False
                                            display_player_win(onevsonewindown, "player2")
                                        if sound_enabled:
                                                sound_file_path = './song/start_song.mp3'
                                                play_sound(sound_file_path)
                                elif event.type ==pygame.MOUSEBUTTONDOWN:
                                    if backmenuonevsone_button.collidepoint(event.pos):
                                        createMenuWindow()
                                    elif nextlevelonevsone_button.collidepoint(event.pos):
                                        nextLevelonevsone()
                                    elif backlevelonevsone_button.collidepoint(event.pos):
                                        backLevelonevsone()
                                    elif resetplayer_button.collidepoint(event.pos):
                                        initsoloLevel()
                                    elif chooselevel_button.collidepoint(event.pos):
                                        createSoloLevelsWindow()
                            print_game_a_b_no_update(game.get_matrix(),onevsonewindown,0,0)
                            pygame.display.update()
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        if flag1==False:
            print_game_a_b_no_update(game.get_matrix(),onevsonewindown,0,0)
            flag1=True
        elif flag2==False:
            print_game_a_b_player2_no_update(game2.get_matrix(),onevsonewindown,a,b)
            flag2=True
        pygame.display.update()   

def initLevel():
    global current_level,node_generated,start,end,solution,moves,sound_enabled,depth_count
    level=current_level
    game = Game(map_open('.\levels',level))
    play_mode_window = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Play Mode")
    algorithms = ["BFS Algorithm", "DFS Algorithm", "UCS Algorithm", "Greedy Algorithm", "A* Algorithm", "Dijkstra Algorithm", "ID Algorithm"]
    current_algorithm_index = 0
    background_image = pygame.image.load("images\game_background.png")
    background_image = pygame.transform.scale(background_image, (800, 600))
    play_mode_window.blit(background_image, (0, 0))
    font_path = './font/fontpixel.ttf'
    font = pygame.font.Font(font_path, 15)
    font2 = pygame.font.Font(font_path, 13)
    draw_speaker(play_mode_window)
    print_game(game.get_matrix(),play_mode_window)
    text = font.render(f"Level: {current_level}", True, (255, 255, 255))
    text_rect = text.get_rect(center=(695, 280))
    play_mode_window.blit(text, text_rect)

    pygame.display.flip()
    image = pygame.image.load("images/menu.png")
    image = pygame.transform.scale(image, (120, 50))
    image2 = pygame.image.load("images/back.png")
    image2 = pygame.transform.scale(image2, (120, 70))
    image3 = pygame.image.load("images/next.png")
    image3 = pygame.transform.scale(image3, (120, 70))
    image4 = pygame.image.load("images/undo.png")
    image4 = pygame.transform.scale(image4, (120, 50))
    image5 = pygame.image.load("images/restart.png")
    image5 = pygame.transform.scale(image5, (120, 60))
    image6 = pygame.image.load("images/solve.png")
    image6 = pygame.transform.scale(image6, (120, 50))
    
    undo_button = pygame.draw.rect(play_mode_window, (64, 224, 208), (650, 420, 100, 30))
    resetlevel_button = pygame.draw.rect(play_mode_window, (64, 224, 208), (650, 370, 100, 30))
    nextlevel_button = pygame.draw.rect(play_mode_window, (64, 224, 208), (650, 470, 100, 30))
    backlevel_button = pygame.draw.rect(play_mode_window, (64, 224, 208), (650, 520, 100, 30))
    backmenu_button = pygame.draw.rect(play_mode_window, (0, 0, 0,0), (70, 525, 100, 30))
    text_box_rect = pygame.Rect(250, 525, 300, 30)
    solve_button_rect = pygame.Rect(650, 310, 100, 30)
    pygame.draw.rect(play_mode_window, (230, 230, 250), text_box_rect)
    text_surface = font.render(algorithms[current_algorithm_index], True, (0, 0, 0))
    text_rect = text_surface.get_rect(center=(text_box_rect.centerx, text_box_rect.centery))
    play_mode_window.blit(text_surface, text_rect.topleft)
    pygame.draw.rect(play_mode_window, (100, 100, 255), solve_button_rect)
    

    play_mode_window.blit(image, (60, 515))
    play_mode_window.blit(image2, (640, 500))
    play_mode_window.blit(image3, (640, 457))
    play_mode_window.blit(image4, (640, 410))
    play_mode_window.blit(image5, (640, 350))
    play_mode_window.blit(image6, (640, 300))
    pygame.display.update()  
    sol = ""
    i = 0
    flagAuto = 0
    while True:
        if sol == "NoSol":
            display_end(play_mode_window,"Cannot")
        if sol == "TimeOut":
            display_end(play_mode_window,"Out")
        if game.is_completed():
            display_end(play_mode_window,"Done")
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    sol = AstarSolution(game)
                    flagAuto = 1
                if event.key == pygame.K_s:
                    bfs_thread = threading.Thread(target=run_BFS_threaded, args=(game,play_mode_window))
                    bfs_thread.start()
                elif event.key == pygame.K_u:
                    undo_move(game,play_mode_window)
                elif event.key==pygame.K_n:
                    nextLevel()
                elif event.key==pygame.K_b:
                    backLevel()
                elif event.key==pygame.K_r:
                    initLevel()
                elif event.key == pygame.K_UP: 
                    game.move(0,-1, True)
                    print_game(game.get_matrix(),play_mode_window)
                elif event.key == pygame.K_DOWN: 
                    game.move(0,1, True)
                    print_game(game.get_matrix(),play_mode_window)
                elif event.key == pygame.K_LEFT: 
                    game.move(-1,0, True)
                    print_game(game.get_matrix(),play_mode_window)
                elif event.key == pygame.K_RIGHT: 
                    game.move(1,0, True)
                    print_game(game.get_matrix(),play_mode_window)
                elif event.key == pygame.K_q: sys.exit(0)
                elif event.key == pygame.K_l: game.unmove()
                elif event.key == pygame.K_c: sol = ""
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if undo_button.collidepoint(event.pos):
                    undo_move(game,play_mode_window)
                elif resetlevel_button.collidepoint(event.pos):
                    initLevel()
                elif nextlevel_button.collidepoint(event.pos):
                    nextLevel()
                elif backlevel_button.collidepoint(event.pos):
                    backLevel()
                elif 20 <= event.pos[0] <= 90 and 20 <= event.pos[1] <= 70:
                    sound_enabled = not sound_enabled
                    if sound_enabled==True:
                        play_sound(sound_file_path)
                        clear_speaker(play_mode_window)
                        draw_speaker(play_mode_window)
                        pygame.display.update()
                    elif sound_enabled==False:
                        stop_sound()
                        clear_speaker(play_mode_window)
                        draw_speaker(play_mode_window)
                        pygame.display.update()
                elif backmenu_button.collidepoint(event.pos):
                    createMenuWindow()
                elif text_box_rect.collidepoint(event.pos):
                        current_algorithm_index = (current_algorithm_index + 1) % len(algorithms)
                        if current_algorithm_index > 7:
                            current_algorithm_index = 0
                        text_box_rect = pygame.Rect(250, 525, 300, 30)
                        pygame.draw.rect(play_mode_window, (230, 230, 250), text_box_rect)
                        text_surface = font.render(algorithms[current_algorithm_index], True, (0, 0, 0))
                        text_rect = text_surface.get_rect(center=(text_box_rect.centerx, text_box_rect.centery))
                        play_mode_window.blit(text_surface, text_rect.topleft)
                        
                elif solve_button_rect.collidepoint(event.pos):
                    algorithm_to_solve = algorithms[current_algorithm_index]
                    if algorithm_to_solve == "BFS Algorithm":
                        node_generated=0
                        start=0
                        end=0
                        solution=None
                        sol = BFSsolution(game)
                        text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
                        text_rect2 = text2.get_rect(topleft=(120, 120))
                        text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
                        text_rect3 = text3.get_rect(topleft=(120, 90))
                        text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
                        text_rect4 = text4.get_rect(topleft=(120, 60))
                        text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
                        text_rect5 = text5.get_rect(topleft=(120, 150))
                        play_mode_window.blit(text5, text_rect5)
                        play_mode_window.blit(text2, text_rect2)
                        play_mode_window.blit(text3, text_rect3)
                        play_mode_window.blit(text4, text_rect4)
                        
                        flagAuto=1
                    elif algorithm_to_solve == "DFS Algorithm":
                        node_generated=0
                        start=0
                        end=0
                        solution=None
                        sol = DFSsolution(game)
                        text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
                        text_rect2 = text2.get_rect(topleft=(120, 120))
                        text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
                        text_rect3 = text3.get_rect(topleft=(120, 90))
                        text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
                        text_rect4 = text4.get_rect(topleft=(120, 60))
                        text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
                        text_rect5 = text5.get_rect(topleft=(120, 150))
                        play_mode_window.blit(text5, text_rect5)
                        play_mode_window.blit(text2, text_rect2)
                        play_mode_window.blit(text3, text_rect3)
                        play_mode_window.blit(text4, text_rect4)
                        flagAuto=1
                    elif algorithm_to_solve == "UCS Algorithm":
                        node_generated=0
                        start=0
                        end=0
                        solution=None
                        sol = UCSsolution(game)
                        text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
                        text_rect2 = text2.get_rect(topleft=(120, 120))
                        text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
                        text_rect3 = text3.get_rect(topleft=(120, 90))
                        text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
                        text_rect4 = text4.get_rect(topleft=(120, 60))
                        text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
                        text_rect5 = text5.get_rect(topleft=(120, 150))
                        play_mode_window.blit(text5, text_rect5)
                        play_mode_window.blit(text2, text_rect2)
                        play_mode_window.blit(text3, text_rect3)
                        play_mode_window.blit(text4, text_rect4)
                        flagAuto=1
                    elif algorithm_to_solve == "Greedy Algorithm":
                        node_generated=0
                        start=0
                        end=0
                        solution=None
                        sol = greedySolution(game)
                        text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
                        text_rect2 = text2.get_rect(topleft=(120, 120))
                        text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
                        text_rect3 = text3.get_rect(topleft=(120, 90))
                        text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
                        text_rect4 = text4.get_rect(topleft=(120, 60))
                        text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
                        text_rect5 = text5.get_rect(topleft=(120, 150))
                        play_mode_window.blit(text5, text_rect5)
                        play_mode_window.blit(text2, text_rect2)
                        play_mode_window.blit(text3, text_rect3)
                        play_mode_window.blit(text4, text_rect4)
                        flagAuto=1
                    elif algorithm_to_solve == "A* Algorithm":
                        node_generated=0
                        start=0
                        end=0
                        solution=None
                        sol = AstarSolution(game)
                        text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
                        text_rect2 = text2.get_rect(topleft=(120, 120))
                        text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
                        text_rect3 = text3.get_rect(topleft=(120, 90))
                        text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
                        text_rect4 = text4.get_rect(topleft=(120, 60))
                        text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
                        text_rect5 = text5.get_rect(topleft=(120, 150))
                        play_mode_window.blit(text5, text_rect5)
                        play_mode_window.blit(text2, text_rect2)
                        play_mode_window.blit(text3, text_rect3)
                        play_mode_window.blit(text4, text_rect4)
                        flagAuto=1
                    elif algorithm_to_solve == "ID Algorithm":
                        node_generated=0
                        start=0
                        end=0
                        solution=None
                        depth_count=0
                        sol = IDSolution(game)
                        text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
                        text_rect2 = text2.get_rect(topleft=(80, 120))
                        text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
                        text_rect3 = text3.get_rect(topleft=(80, 90))
                        text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
                        text_rect4 = text4.get_rect(topleft=(80, 60))
                        text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
                        text_rect5 = text5.get_rect(topleft=(80, 150))
                        text6 = font2.render(f"Depth Limit: {depth_count}", True,  (249, 244 ,0))
                        text_rect6 = text6.get_rect(topleft=(80, 180))
                        play_mode_window.blit(text5, text_rect5)
                        play_mode_window.blit(text2, text_rect2)
                        play_mode_window.blit(text3, text_rect3)
                        play_mode_window.blit(text4, text_rect4)
                        play_mode_window.blit(text6, text_rect6)
                        flagAuto=1
                    elif algorithm_to_solve == "Dijkstra Algorithm":
                        node_generated=0
                        start=0
                        end=0
                        solution=None
                        sol = DijkstraSolution(game)
                        text2 = font2.render(f"Visited Nodes: {node_generated}", True, (249, 244 ,0))
                        text_rect2 = text2.get_rect(topleft=(120, 120))
                        text3 = font2.render(f"Time Execute: {round(end-start,2)}", True,  (249, 244 ,0))
                        text_rect3 = text3.get_rect(topleft=(120, 90))
                        text4 = font2.render(f"Solution: {solution}", True,  (249, 244 ,0))
                        text_rect4 = text4.get_rect(topleft=(120, 60))
                        text5 = font2.render(f"Moves: {moves}", True,  (249, 244 ,0))
                        text_rect5 = text5.get_rect(topleft=(120, 150))
                        play_mode_window.blit(text5, text_rect5)
                        play_mode_window.blit(text2, text_rect2)
                        play_mode_window.blit(text3, text_rect3)
                        play_mode_window.blit(text4, text_rect4)
                        flagAuto=1
                        
                    
            elif event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if (flagAuto) and (i < len(sol)):
            playByBot(game,sol[i],play_mode_window)
            i += 1
            if i == len(sol): flagAuto = 0
            time.sleep(0.05)

        pygame.display.update()

def createPlayModeWindow():
    initLevel()
    
wall = pygame.image.load('.\images\wall.png')
box = pygame.image.load('.\images/box.png')
box_docked = pygame.image.load('.\images/box_on_target.png')
worker = pygame.image.load('.\images\player.png')
worker2 = pygame.image.load('.\images\player2.png')
worker_docked = pygame.image.load('.\images\player.png')
worker_docked2 = pygame.image.load('.\images\player2.png')
docker = pygame.image.load('.\images/target.png')
background = 255, 255, 255


def playByBot(game,move,screen):
    if move == "U":
        game.move(0,-1,False)
   
    elif move == "D":
        game.move(0,1,False)
   
    elif move == "L":
        game.move(-1,0,False)
   
    elif move == "R":
        game.move(1,0,False)
        
    else:
        game.move(0,0,False)
    print_game(game.get_matrix(), screen)
    pygame.time.delay(100)
def playByBotPlayer1(game,move,screen):
    if move == "U":
        game.move(0,-1,False)
   
    elif move == "D":
        game.move(0,1,False)
   
    elif move == "L":
        game.move(-1,0,False)
   
    elif move == "R":
        game.move(1,0,False)
        
    else:
        game.move(0,0,False)
    print_game_a_b_player2_no_update(game.get_matrix(), screen,600,0)
    pygame.display.update()
    pygame.time.delay(300)
def playByBotPlayer2(game,move,screen):
    if move == "U":
        game.move(0,-1,False)
   
    elif move == "D":
        game.move(0,1,False)
   
    elif move == "L":
        game.move(-1,0,False)
   
    elif move == "R":
        game.move(1,0,False)
        
    else:
        game.move(0,0,False)
    print_game_a_b_no_update(game.get_matrix(), screen,0,0)
    pygame.display.update()
    pygame.time.delay(300)

def map_open(filename, level):
    matrix = []
#   if level < 1 or level > 50:
    if int(level) < 1:
        print("ERROR: Level "+str(level)+" is out of range")
        sys.exit(1)
    else:
        file = open(filename,'r')
        level_found = False
        for line in file:
            row = []
            if not level_found:
                if  "Level "+str(level) == line.strip():
                    level_found = True
            else:
                if line.strip() != "":
                    row = []
                    for c in line:
                        if c != '\n' and c in [' ','#','@','+','$','*','.']:
                            row.append(c)
                        elif c == '\n': #jump to next row when newline
                            continue
                        else:
                            print("ERROR: Level "+str(level)+" has invalid value "+c)
                            sys.exit(1)
                    matrix.append(row)
                else:
                    break
        return matrix
def print_game_a_b(matrix, screen, start_a, start_b):
    background_color = (255, 255, 255)
    map_width = len(matrix[0]) * 36
    map_height = len(matrix) * 36

    cell_size = 36
    x = start_a
    y = start_b
    pygame.draw.rect(screen, background_color, (x, y, map_width, map_height))
    for row in matrix:
        for char in row:
            if char == '#':  # wall
                screen.blit(wall, (x, y))
            elif char == '@':  # worker on floor
                screen.blit(worker, (x, y))
            elif char == '.':  # dock
                screen.blit(docker, (x, y))
            elif char == '*':  # box on dock
                screen.blit(box_docked, (x, y))
            elif char == '$':  # box
                screen.blit(box, (x, y))
            elif char == '+':  # worker on dock
                screen.blit(worker_docked, (x, y))
            x += cell_size
        x = start_a
        y += cell_size
        pygame.display.update()
def print_game_a_b_no_update(matrix, screen, start_a, start_b):
    background_color = (255, 255, 255)
    map_width = len(matrix[0]) * 36
    map_height = len(matrix) * 36

    cell_size = 36
    x = start_a
    y = start_b
    pygame.draw.rect(screen, background_color, (x, y, map_width, map_height))
    for row in matrix:
        for char in row:
            if char == '#':  # wall
                screen.blit(wall, (x, y))
            elif char == '@':  # worker on floor
                screen.blit(worker, (x, y))
            elif char == '.':  # dock
                screen.blit(docker, (x, y))
            elif char == '*':  # box on dock
                screen.blit(box_docked, (x, y))
            elif char == '$':  # box
                screen.blit(box, (x, y))
            elif char == '+':  # worker on dock
                screen.blit(worker_docked, (x, y))
            x += cell_size
        x = start_a
        y += cell_size
       

def print_game_a_b_player2(matrix, screen, start_a, start_b):
    background_color = (255, 255, 255)
    map_width = len(matrix[0]) * 36
    map_height = len(matrix) * 36

    cell_size = 36
    x = start_a
    y = start_b
    pygame.draw.rect(screen, background_color, (x, y, map_width, map_height))
    for row in matrix:
        for char in row:
            if char == '#':  # wall
                screen.blit(wall, (x, y))
            elif char == '@':  # worker on floor
                screen.blit(worker2, (x, y))
            elif char == '.':  # dock
                screen.blit(docker, (x, y))
            elif char == '*':  # box on dock
                screen.blit(box_docked, (x, y))
            elif char == '$':  # box
                screen.blit(box, (x, y))
            elif char == '+':  # worker on dock
                screen.blit(worker_docked2, (x, y))
            x += cell_size
        x = start_a
        y += cell_size
        pygame.display.update()
def print_game_a_b_player2_no_update(matrix, screen, start_a, start_b):
    background_color = (255, 255, 255)
    map_width = len(matrix[0]) * 36
    map_height = len(matrix) * 36

    cell_size = 36
    x = start_a
    y = start_b
    pygame.draw.rect(screen, background_color, (x, y, map_width, map_height))
    for row in matrix:
        for char in row:
            if char == '#':  # wall
                screen.blit(wall, (x, y))
            elif char == '@':  # worker on floor
                screen.blit(worker2, (x, y))
            elif char == '.':  # dock
                screen.blit(docker, (x, y))
            elif char == '*':  # box on dock
                screen.blit(box_docked, (x, y))
            elif char == '$':  # box
                screen.blit(box, (x, y))
            elif char == '+':  # worker on dock
                screen.blit(worker_docked2, (x, y))
            x += cell_size
        x = start_a
        y += cell_size


def print_game(matrix, screen):
    matrix_height = len(matrix)
    matrix_width = len(matrix[0])
    background_color = (255, 255, 255)

    cell_size = 36
    map_width = len(matrix[0]) * 36
    map_height = len(matrix) * 36

    # Tính toán vị trí bắt đầu để căn giữa màn hình
    start_x = (800 - matrix_width * cell_size) // 2
    start_y = (600 - matrix_height * cell_size) // 2
    pygame.draw.rect(screen, background_color, (start_x, start_y, map_width, map_height))
    x = start_x
    y = start_y
    for row in matrix:
        for char in row:
            if char == '#':  # wall
                screen.blit(wall, (x, y))
            elif char == '@':  # worker on floor
                screen.blit(worker, (x, y))
            elif char == '.':  # dock
                screen.blit(docker, (x, y))
            elif char == '*':  # box on dock
                screen.blit(box_docked, (x, y))
            elif char == '$':  # box
                screen.blit(box, (x, y))
            elif char == '+':  # worker on dock
                screen.blit(worker_docked, (x, y))
            x += cell_size
        x = start_x
        y += cell_size
    pygame.display.update()

def get_key():
  while 1:
    event = pygame.event.poll()
    if event.type == pygame.KEYDOWN:
      return event.key
    else:
      pass

def display_box(screen, message):
  fontobject = pygame.font.Font(None,18)
  pygame.draw.rect(screen, (0,0,0),
                   ((screen.get_width() / 2) - 100,
                    (screen.get_height() / 2) - 10,
                    200,20), 0)
  pygame.draw.rect(screen, (255,255,255),
                   ((screen.get_width() / 2) - 102,
                    (screen.get_height() / 2) - 12,
                    204,24), 1)
  if len(message) != 0:
    screen.blit(fontobject.render(message, 1, (255,255,255)),
                ((screen.get_width() / 2) - 100, (screen.get_height() / 2) - 10))
  pygame.display.flip()

def display_end(screen, msg):
    if msg == "Done":
        message = " Level Completed  "
    elif msg == "Cannot":
        message = "No Solution"
    elif msg == "Out":
        message = "Time Out! Cannot find solution"

    font_object = pygame.font.Font(None, 30)
    text_surface = font_object.render(message, True, (255, 255, 255))

    # Calculate the position to center the text
    text_rect = text_surface.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2))

    # Draw the background rectangle
    pygame.draw.rect(screen, (245, 168, 154), text_rect, 0)

    # Draw the border rectangle
    pygame.draw.rect(screen, (0, 0, 0), text_rect, 1)

    # Blit the text surface onto the screen
    screen.blit(text_surface, text_rect.topleft)

    pygame.display.flip()
def display_player_win(screen, msg):
    if msg == "player1":
        message = "Player 1 Winner !  "
    elif msg == "player2":
        message = " Player 2 Winner ! "
    elif msg == "draw":
        message = "Draw"
    elif msg=="AI":
        message="AI Win !!!"
    font_object = pygame.font.Font(None, 30)
    text_surface = font_object.render(message, True, (255, 255, 255))

    # Calculate the position to center the text
    text_rect = text_surface.get_rect(center=(screen.get_width() / 2, screen.get_height() / 2))

    # Draw the background rectangle
    pygame.draw.rect(screen, (245, 168, 154), text_rect, 0)

    # Draw the border rectangle
    pygame.draw.rect(screen, (0, 0, 0), text_rect, 1)

    # Blit the text surface onto the screen
    screen.blit(text_surface, text_rect.topleft)

    pygame.display.flip()



def ask(screen, question):
  "ask(screen, question) -> answer"
  pygame.font.init()
  current_string = []
  display_box(screen, question + ": " + "".join(current_string))
  while 1:
    inkey = get_key()
    if inkey == pygame.K_BACKSPACE:
      current_string = current_string[0:-1]
    elif inkey == pygame.K_RETURN:
      break
    elif inkey == pygame.K_MINUS:
      current_string.append("_")
    elif inkey <= 127:
      current_string.append(chr(inkey))
    display_box(screen, question + ": " + "".join(current_string))
  return "".join(current_string)
def main():
    pygame.init()
    pygame.display.set_caption("Sokoban Game")
    createWelcomWindow()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pygame.display.update()

    pygame.quit()

if __name__ == "__main__":
    main()