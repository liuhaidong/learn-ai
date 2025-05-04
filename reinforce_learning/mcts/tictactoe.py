import numpy as np
import math

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 0:空, 1:X, 2:O
        self.current_player = 1       # X先手

    def get_valid_moves(self):
        """返回所有可落子的位置"""
        return [(i, j) for i in range(3) for j in range(3) if self.board[i, j] == 0]

    def make_move(self, move):
        """执行落子"""
        i, j = move
        if self.board[i, j] == 0:
            self.board[i, j] = self.current_player
            self.current_player = 3 - self.current_player  # 切换玩家（1→2或2→1）
            return True
        return False

    def check_winner(self):
        """检查是否有人获胜"""
        for player in [1, 2]:
            # 检查行和列
            for i in range(3):
                if all(self.board[i, :] == player) or all(self.board[:, i] == player):
                    return player
            # 检查对角线
            if (self.board[0, 0] == player and self.board[1, 1] == player and self.board[2, 2] == player) or \
               (self.board[0, 2] == player and self.board[1, 1] == player and self.board[2, 0] == player):
                return player
        return 0  # 无胜负

    def is_game_over(self):
        """游戏是否结束"""
        return self.check_winner() != 0 or len(self.get_valid_moves()) == 0

    def copy(self):
        """复制当前游戏状态"""
        new_game = TicTacToe()
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        return new_game
    
class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state      # 当前游戏状态
        self.parent = parent              # 父节点
        self.move = move                  # 导致此节点的动作
        self.children = []                # 子节点列表
        self.visits = 0                   # 访问次数
        self.wins = 0                     # 累计奖励（胜利次数）
    
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
        for move in self.game_state.get_valid_moves():
            new_game = self.game_state.copy()
            new_game.make_move(move)
            self.children.append(MCTSNode(new_game, parent=self, move=move))
    
    def simulate(self):
        """随机模拟到游戏结束"""
        game = self.game_state.copy()
        while not game.is_game_over():
            valid_moves = game.get_valid_moves()
            move = valid_moves[np.random.choice(len(valid_moves))]
            game.make_move(move)
        winner = game.check_winner()
        # 假设当前节点是AI（玩家2），胜利返回1，失败返回0
        return 1 if winner == 2 else 0
    
class MCTS:
    def __init__(self, iterations=1000):
        self.iterations = iterations  # 搜索迭代次数
    
    def search(self, initial_state):
        root = MCTSNode(initial_state)
        
        for _ in range(self.iterations):
            # 选择阶段
            node = root
            while node.children:
                node = node.select_child()
            
            # 扩展阶段
            if not node.game_state.is_game_over():
                node.expand()
                if node.children:
                    node = np.random.choice(node.children)
            
            # 模拟阶段
            result = node.simulate()
            
            # 回溯阶段
            while node is not None:
                node.visits += 1
                node.wins += result
                node = node.parent
        
        # 选择访问次数最多的子节点
        best_move = max(root.children, key=lambda child: child.visits).move
        return best_move

def play_game():
    game = TicTacToe()
    mcts = MCTS(iterations=1000)
    
    while not game.is_game_over():
        print(game.board)
        if game.current_player == 1:  # 人类玩家（X）
            row, col = map(int, input("输入你的落子（行 列，如0 0）: ").split())
            game.make_move((row, col))
        else:  # AI玩家（O）
            print("AI思考中...")
            best_move = mcts.search(game)
            game.make_move(best_move)
    
    print("最终棋盘:")
    print(game.board)
    winner = game.check_winner()
    print("X获胜!" if winner == 1 else "O获胜!" if winner == 2 else "平局!")

# 开始游戏
play_game()