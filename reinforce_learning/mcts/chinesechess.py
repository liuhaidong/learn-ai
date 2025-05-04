import numpy as np
import math
import random
from collections import defaultdict

class ChineseChess:
    # 棋子类型
    EMPTY = 0
    R_GENERAL = 1    # 红帅
    R_ADVISOR = 2     # 红仕
    R_ELEPHANT = 3    # 红相
    R_HORSE = 4       # 红马
    R_CHARIOT = 5     # 红车
    R_CANNON = 6      # 红炮
    R_SOLDIER = 7     # 红兵
    B_GENERAL = 8     # 黑将
    B_ADVISOR = 9     # 黑士
    B_ELEPHANT = 10   # 黑象
    B_HORSE = 11      # 黑马
    B_CHARIOT = 12    # 黑车
    B_CANNON = 13     # 黑炮
    B_SOLDIER = 14    # 黑卒
    
    def __init__(self):
        # 初始化棋盘
        self.board = np.zeros((10, 9), dtype=int)
        self.current_player = 1  # 1: 红方, -1: 黑方
        self.game_over = False
        self.winner = None
        self.last_move = None
        self._initialize_board()
    
    def _initialize_board(self):
        # 初始化棋盘布局
        # 黑方(上方)
        self.board[0] = [
            self.B_CHARIOT, self.B_HORSE, self.B_ELEPHANT, self.B_ADVISOR, 
            self.B_GENERAL, self.B_ADVISOR, self.B_ELEPHANT, self.B_HORSE, self.B_CHARIOT
        ]
        self.board[2][1] = self.B_CANNON
        self.board[2][7] = self.B_CANNON
        self.board[3] = [self.B_SOLDIER if i % 2 == 0 else self.EMPTY for i in range(9)]
        
        # 红方(下方)
        self.board[9] = [
            self.R_CHARIOT, self.R_HORSE, self.R_ELEPHANT, self.R_ADVISOR, 
            self.R_GENERAL, self.R_ADVISOR, self.R_ELEPHANT, self.R_HORSE, self.R_CHARIOT
        ]
        self.board[7][1] = self.R_CANNON
        self.board[7][7] = self.R_CANNON
        self.board[6] = [self.R_SOLDIER if i % 2 == 0 else self.EMPTY for i in range(9)]
    
    def get_valid_moves(self):
        """获取当前玩家的所有合法走法"""
        moves = []
        for i in range(10):
            for j in range(9):
                piece = self.board[i][j]
                if (self.current_player == 1 and 1 <= piece <= 7) or \
                   (self.current_player == -1 and 8 <= piece <= 14):
                    moves.extend(self._get_moves_for_piece(i, j))
        return moves
    
    def _get_moves_for_piece(self, x, y):
        """获取特定棋子的所有合法走法"""
        piece = self.board[x][y]
        moves = []
        
        if piece in [self.R_GENERAL, self.B_GENERAL]:  # 将/帅
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self._is_valid_move(x, y, nx, ny, piece):
                    moves.append((x, y, nx, ny))
            # 将帅对面特殊规则
            if piece == self.R_GENERAL:
                for i in range(x-1, -1, -1):
                    if self.board[i][y] != self.EMPTY:
                        if self.board[i][y] == self.B_GENERAL:
                            moves.append((x, y, i, y))
                        break
            elif piece == self.B_GENERAL:
                for i in range(x+1, 10):
                    if self.board[i][y] != self.EMPTY:
                        if self.board[i][y] == self.R_GENERAL:
                            moves.append((x, y, i, y))
                        break
        
        elif piece in [self.R_ADVISOR, self.B_ADVISOR]:  # 仕/士
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self._is_valid_move(x, y, nx, ny, piece):
                    moves.append((x, y, nx, ny))
        
        elif piece in [self.R_ELEPHANT, self.B_ELEPHANT]:  # 相/象
            directions = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                bx, by = x + dx//2, y + dy//2  # 象眼位置
                if self._is_valid_move(x, y, nx, ny, piece) and self.board[bx][by] == self.EMPTY:
                    moves.append((x, y, nx, ny))
        
        elif piece in [self.R_HORSE, self.B_HORSE]:  # 马
            horse_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
                          (1, -2), (1, 2), (2, -1), (2, 1)]
            for dx, dy in horse_moves:
                nx, ny = x + dx, y + dy
                # 马腿检查
                if abs(dx) == 2 and self.board[x + dx//2][y] != self.EMPTY:
                    continue
                if abs(dy) == 2 and self.board[x][y + dy//2] != self.EMPTY:
                    continue
                if self._is_valid_move(x, y, nx, ny, piece):
                    moves.append((x, y, nx, ny))
        
        elif piece in [self.R_CHARIOT, self.B_CHARIOT]:  # 车
            # 横向移动
            for dy in [-1, 1]:
                ny = y + dy
                while 0 <= ny < 9:
                    if self.board[x][ny] == self.EMPTY:
                        moves.append((x, y, x, ny))
                    else:
                        if self._can_capture(piece, self.board[x][ny]):
                            moves.append((x, y, x, ny))
                        break
                    ny += dy
            # 纵向移动
            for dx in [-1, 1]:
                nx = x + dx
                while 0 <= nx < 10:
                    if self.board[nx][y] == self.EMPTY:
                        moves.append((x, y, nx, y))
                    else:
                        if self._can_capture(piece, self.board[nx][y]):
                            moves.append((x, y, nx, y))
                        break
                    nx += dx
        
        elif piece in [self.R_CANNON, self.B_CANNON]:  # 炮
            # 横向移动
            for dy in [-1, 1]:
                ny = y + dy
                has_piece = False
                while 0 <= ny < 9:
                    if not has_piece:
                        if self.board[x][ny] == self.EMPTY:
                            moves.append((x, y, x, ny))
                        else:
                            has_piece = True
                    else:
                        if self.board[x][ny] != self.EMPTY:
                            if self._can_capture(piece, self.board[x][ny]):
                                moves.append((x, y, x, ny))
                            break
                    ny += dy
            # 纵向移动
            for dx in [-1, 1]:
                nx = x + dx
                has_piece = False
                while 0 <= nx < 10:
                    if not has_piece:
                        if self.board[nx][y] == self.EMPTY:
                            moves.append((x, y, nx, y))
                        else:
                            has_piece = True
                    else:
                        if self.board[nx][y] != self.EMPTY:
                            if self._can_capture(piece, self.board[nx][y]):
                                moves.append((x, y, nx, y))
                            break
                    nx += dx
        
        elif piece in [self.R_SOLDIER, self.B_SOLDIER]:  # 兵/卒
            if piece == self.R_SOLDIER:  # 红兵
                if x > 0:  # 可以向前
                    if self._is_valid_move(x, y, x-1, y, piece):
                        moves.append((x, y, x-1, y))
                if x <= 4:  # 过河后可以左右移动
                    if y > 0 and self._is_valid_move(x, y, x, y-1, piece):
                        moves.append((x, y, x, y-1))
                    if y < 8 and self._is_valid_move(x, y, x, y+1, piece):
                        moves.append((x, y, x, y+1))
            else:  # 黑卒
                if x < 9:  # 可以向前
                    if self._is_valid_move(x, y, x+1, y, piece):
                        moves.append((x, y, x+1, y))
                if x >= 5:  # 过河后可以左右移动
                    if y > 0 and self._is_valid_move(x, y, x, y-1, piece):
                        moves.append((x, y, x, y-1))
                    if y < 8 and self._is_valid_move(x, y, x, y+1, piece):
                        moves.append((x, y, x, y+1))
        
        return moves
    
    def _is_valid_move(self, x, y, nx, ny, piece):
        """检查移动是否基本有效"""
        # 检查是否在棋盘内
        if not (0 <= nx < 10 and 0 <= ny < 9):
            return False
        
        # 检查目标位置是否是自己的棋子
        target = self.board[nx][ny]
        if target != self.EMPTY:
            if (self.current_player == 1 and 1 <= target <= 7) or \
               (self.current_player == -1 and 8 <= target <= 14):
                return False
        
        # 特殊规则检查
        if piece in [self.R_GENERAL, self.B_GENERAL]:  # 将/帅
            # 将帅必须在九宫格内
            if (piece == self.R_GENERAL and not (7 <= nx <= 9 and 3 <= ny <= 5)) or \
               (piece == self.B_GENERAL and not (0 <= nx <= 2 and 3 <= ny <= 5)):
                return False
        
        elif piece in [self.R_ADVISOR, self.B_ADVISOR]:  # 仕/士
            # 士必须在九宫格内且走斜线
            if (piece == self.R_ADVISOR and not (7 <= nx <= 9 and 3 <= ny <= 5)) or \
               (piece == self.B_ADVISOR and not (0 <= nx <= 2 and 3 <= ny <= 5)):
                return False
        
        elif piece in [self.R_ELEPHANT, self.B_ELEPHANT]:  # 相/象
            # 相/象不能过河
            if (piece == self.R_ELEPHANT and nx < 5) or \
               (piece == self.B_ELEPHANT and nx > 4):
                return False
        
        return True
    
    def _can_capture(self, attacker, defender):
        """检查是否可以吃子"""
        if defender == self.EMPTY:
            return False
        # 红方棋子(1-7)不能吃红方棋子，黑方棋子(8-14)不能吃黑方棋子
        if (1 <= attacker <= 7 and 1 <= defender <= 7) or \
           (8 <= attacker <= 14 and 8 <= defender <= 14):
            return False
        return True
    
    def make_move(self, move):
        """执行走法"""
        x, y, nx, ny = move
        self.board[nx][ny] = self.board[x][y]
        self.board[x][y] = self.EMPTY
        self.last_move = move
        self.current_player *= -1  # 切换玩家
        
        # 检查游戏是否结束
        if self._is_general_captured() or not self._has_valid_moves():
            self.game_over = True
            self.winner = -self.current_player  # 当前玩家无法移动，对手获胜
    
    def _is_general_captured(self):
        """检查将/帅是否被吃"""
        red_general_exists = any(self.R_GENERAL in row for row in self.board)
        black_general_exists = any(self.B_GENERAL in row for row in self.board)
        return not red_general_exists or not black_general_exists
    
    def _has_valid_moves(self):
        """当前玩家是否有合法走法"""
        return len(self.get_valid_moves()) > 0
    
    def is_game_over(self):
        """游戏是否结束"""
        return self.game_over
    
    def get_winner(self):
        """获取胜利者"""
        return self.winner
    
    def copy(self):
        """复制当前游戏状态"""
        new_game = ChineseChess()
        new_game.board = np.copy(self.board)
        new_game.current_player = self.current_player
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.last_move = self.last_move
        return new_game
    
class MCTSNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.untried_moves = None
    
    def ucb_score(self, exploration=1.414):
        """计算UCB1分数"""
        if self.visits == 0:
            return float('inf')
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
            moves = game.get_valid_moves()
            if not moves:
                break
            move = random.choice(moves)
            game.make_move(move)
        
        winner = game.get_winner()
        if winner == 1:  # 红方胜
            return 1 if self.game_state.current_player == 1 else 0
        elif winner == -1:  # 黑方胜
            return 1 if self.game_state.current_player == -1 else 0
        return 0.5  # 平局
    
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
        
        if not root.children:
            return None
        
        # 选择访问次数最多的子节点
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.move
    
def print_board(board):
    """打印棋盘"""
    piece_symbols = {
        ChineseChess.EMPTY: '·',
        ChineseChess.R_GENERAL: '帅', ChineseChess.B_GENERAL: '将',
        ChineseChess.R_ADVISOR: '仕', ChineseChess.B_ADVISOR: '士',
        ChineseChess.R_ELEPHANT: '相', ChineseChess.B_ELEPHANT: '象',
        ChineseChess.R_HORSE: '马', ChineseChess.B_HORSE: '马',
        ChineseChess.R_CHARIOT: '车', ChineseChess.B_CHARIOT: '车',
        ChineseChess.R_CANNON: '炮', ChineseChess.B_CANNON: '炮',
        ChineseChess.R_SOLDIER: '兵', ChineseChess.B_SOLDIER: '卒'
    }
    
    print("   " + " ".join(f"{i}" for i in range(9)))
    for i in range(10):
        row = [piece_symbols[piece] for piece in board[i]]
        print(f"{i} " + " ".join(row))
    print()

def play_game():
    game = ChineseChess()
    mcts = MCTS(iterations=1000)
    
    while not game.is_game_over():
        print("\n当前棋盘:")
        print_board(game.board)
        print(f"当前玩家: {'红方' if game.current_player == 1 else '黑方'}")
        
        if game.current_player == 1:  # 人类玩家(红方)
            while True:
                try:
                    move = input("输入你的走法(如: 7 0 8 0 表示从7,0移动到8,0): ").split()
                    if len(move) != 4:
                        raise ValueError
                    x, y, nx, ny = map(int, move)
                    if (x, y, nx, ny) in game.get_valid_moves():
                        break
                    print("无效的走法，请重新输入！")
                except ValueError:
                    print("输入格式错误，请按'x y nx ny'格式输入数字！")
            
            game.make_move((x, y, nx, ny))
        else:  # AI玩家(黑方)
            print("AI思考中...")
            best_move = mcts.search(game)
            if best_move:
                game.make_move(best_move)
                x, y, nx, ny = best_move
                print(f"AI走法: {x}{y} -> {nx}{ny}")
            else:
                print("AI无法找到合法走法！")
                break
    
    print("\n最终棋盘:")
    print_board(game.board)
    winner = game.get_winner()
    print("红方获胜!" if winner == 1 else "黑方获胜!" if winner == -1 else "平局!")

if __name__ == "__main__":
    print("中国象棋游戏 - 红方先行")
    print("输入格式: x y nx ny (例如: 7 0 8 0 表示从第7行第0列移动到第8行第0列)")
    play_game()