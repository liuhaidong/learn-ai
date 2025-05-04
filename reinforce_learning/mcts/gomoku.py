import numpy as np
import math
import random
from collections import defaultdict

class Gomoku:
    def __init__(self, board_size=10, win_length=5):
        self.board_size = board_size
        self.win_length = win_length
        self.board = np.zeros((board_size, board_size))  # 0: 空, 1: 黑棋, 2: 白棋
        self.current_player = 1  # 黑棋先行
        self.last_move = None

    def get_valid_moves(self):
        """返回所有合法的落子位置"""
        if self.last_move is None:
            # 第一手可以下在任意位置
            return [(i, j) for i in range(self.board_size) for j in range(self.board_size)]
        
        # 为了提高效率，只考虑已有棋子周围的空位
        valid_moves = set()
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        
        # 检查所有已有棋子周围的位置
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] != 0:
                    for di, dj in directions:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.board_size and 0 <= nj < self.board_size and self.board[ni, nj] == 0:
                            valid_moves.add((ni, nj))
        
        return list(valid_moves)

    def make_move(self, move):
        """执行落子"""
        i, j = move
        if self.board[i, j] == 0:
            self.board[i, j] = self.current_player
            self.last_move = (i, j)
            self.current_player = 3 - self.current_player  # 切换玩家
            return True
        return False

    def check_winner(self):
        """检查是否有玩家获胜"""
        if self.last_move is None:
            return 0
        
        i, j = self.last_move
        player = self.board[i, j]
        
        # 检查四个方向
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        
        for di, dj in directions:
            count = 1  # 当前位置已经有一个棋子
            
            # 正向检查
            ni, nj = i + di, j + dj
            while 0 <= ni < self.board_size and 0 <= nj < self.board_size and self.board[ni, nj] == player:
                count += 1
                ni += di
                nj += dj
            
            # 反向检查
            ni, nj = i - di, j - dj
            while 0 <= ni < self.board_size and 0 <= nj < self.board_size and self.board[ni, nj] == player:
                count += 1
                ni -= di
                nj -= dj
            
            if count >= self.win_length:
                return player
        
        return 0  # 无胜负

    def is_game_over(self):
        """游戏是否结束"""
        return self.check_winner() != 0 or len(self.get_valid_moves()) == 0

    def copy(self):
        """复制当前游戏状态"""
        new_game = Gomoku(self.board_size, self.win_length)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.last_move = self.last_move
        return new_game

class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state  # 当前游戏状态
        self.parent = parent          # 父节点
        self.move = move              # 导致此节点的动作
        self.children = []            # 子节点列表
        self.visits = 0               # 访问次数
        self.wins = 0                 # 累计奖励
        self.untried_moves = None     # 未尝试的动作
    
    def ucb_score(self, exploration=1.414):
        """计算UCB1分数"""
        if self.visits == 0:
            return float('inf')  # 未探索的节点优先访问
        return (self.wins / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def select_child(self):
        """选择UCB1分数最高的子节点"""
        return max(self.children, key=lambda child: child.ucb_score())
    
    def expand(self):
        """扩展新节点"""
        if self.untried_moves is None:
            self.untried_moves = self.game_state.get_valid_moves()
        
        if self.untried_moves:
            move = self.untried_moves.pop()
            new_game = self.game_state.copy()
            new_game.make_move(move)
            new_node = MCTSNode(new_game, parent=self, move=move)
            self.children.append(new_node)
            return new_node
        return None
    
    def simulate(self):
        """随机模拟到游戏结束"""
        game = self.game_state.copy()
        while not game.is_game_over():
            valid_moves = game.get_valid_moves()
            move = random.choice(valid_moves)
            game.make_move(move)
        
        winner = game.check_winner()
        # 假设当前节点是AI（玩家2），胜利返回1，失败返回0
        return 1 if winner == 2 else 0
    
    def backpropagate(self, result):
        """回溯更新节点统计信息"""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)

class MCTS:
    def __init__(self, iterations=1000, exploration=1.414):
        self.iterations = iterations
        self.exploration = exploration
    
    def search(self, initial_state):
        root = MCTSNode(initial_state)
        
        for _ in range(self.iterations):
            node = root
            
            # 选择阶段
            while node.children:
                node = node.select_child()
            
            # 扩展阶段
            if not node.game_state.is_game_over():
                expanded_node = node.expand()
                if expanded_node is not None:
                    node = expanded_node
            
            # 模拟阶段
            result = node.simulate()
            
            # 回溯阶段
            node.backpropagate(result)
        
        # 选择访问次数最多的子节点
        if not root.children:
            return None
        
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move

def print_board(board):
    """打印棋盘"""
    print("   " + " ".join(f"{i:2}" for i in range(len(board))))
    for i, row in enumerate(board):
        print(f"{i:2} " + " ".join("○" if cell == 1 else "●" if cell == 2 else "·" for cell in row))

def play_game():
    board_size = 10
    game = Gomoku(board_size=board_size)
    mcts = MCTS(iterations=1000)
    
    while not game.is_game_over():
        print("\n当前棋盘:")
        print_board(game.board)
        
        if game.current_player == 1:  # 人类玩家（黑棋）
            while True:
                try:
                    move = input("输入你的落子（行 列，如4 5）: ").split()
                    if len(move) != 2:
                        raise ValueError
                    row, col = map(int, move)
                    if 0 <= row < board_size and 0 <= col < board_size and game.board[row, col] == 0:
                        break
                    print("无效的落子，请重新输入！")
                except ValueError:
                    print("输入格式错误，请按'行 列'格式输入数字！")
            
            game.make_move((row, col))
        else:  # AI玩家（白棋）
            print("AI思考中...")
            best_move = mcts.search(game)
            if best_move:
                game.make_move(best_move)
                print(f"AI落子于: {best_move}")
            else:
                print("AI无法找到合法落子位置！")
                break
    
    print("\n最终棋盘:")
    print_board(game.board)
    winner = game.check_winner()
    print("黑棋获胜!" if winner == 1 else "白棋获胜!" if winner == 2 else "平局!")

if __name__ == "__main__":
    print("10x10五子棋游戏 - 黑棋(○)先行")
    print("输入格式: 行 列 (例如: 4 5 表示第4行第5列)")
    play_game()