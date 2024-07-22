

import hashlib

class Block():
    def __init__(self, timestamp, transactions, previous_hash):
        """Returns a new Block object. Each block is "chained" to its previous by calling its unique hash
        """
        self.timestamp = timestamp # 产生的时间戳
        self.transactions = transactions
        self.index = 0
        self.previous_hash = previous_hash
        self.miner = None
        self.hash = self.get_hash()
    def get_hash(self):
        """Creates the unique hash for the block. It uses sha256."""
        sha = hashlib.sha256()
        sha.update((str(self.timestamp) + str(self.transactions) + str(self.previous_hash)).encode('utf-8'))
        return sha.hexdigest()

class Blockchain():
    CONSENSUS_POW = 'PoW'
    CONSENSUS_POS = 'PoS'
    def __init__(self, cur_time, mine_time_threshold = 10, transaction_threshold = 10, consensus = CONSENSUS_POS):
        self.all_transactions = []
        self.last_update_time = cur_time
        self.chain = [self.create_genesis_block()]
        self.mine_time_threshold = mine_time_threshold
        self.transaction_threshold = transaction_threshold
        self.consensus_type = consensus
        print("区块链初始化完成, 共识机制为: ", consensus)
        self.to_mine_blocks = []
        self.total_transaction_num = 0
    
    @property
    def length(self):
        return len(self.chain)
    @property
    def transaction_num(self):
        return len(self.all_transactions)

    def create_genesis_block(self):
        return Block(self.last_update_time, [], "0")

    def add_block(self, block, miner):
        assert block in self.to_mine_blocks
        self.to_mine_blocks.remove(block)
        block.miner = miner
        block.index = len(self.chain)
        block.previous_hash = self.chain[-1].hash
        self.chain.append(block)

    def validate_chain(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            if(current.hash != current.get_hash()):
                print("当前区块记录的hash值不等于当前区块的hash值")
                return False
            if(current.previous_hash != previous.get_hash()):
                print("当前区块记录的前一个区块的hash值不等于前一个区块的hash值")
                return False
        return True

    def add_new_transaction(self, transaction):
        self.all_transactions.append(transaction)
        self.total_transaction_num += 1

        
    def generate_to_mine_blocks(self, cur_time):
        # 先判断self.transactions 是否达到阈值,while循环
        while len(self.all_transactions) >= self.transaction_threshold:
            tmp_transactions = self.all_transactions[:self.transaction_threshold]
            # 生成新的区块
            new_block = Block(cur_time, tmp_transactions, self.chain[-1].hash)
            # 将新的区块加入待挖矿区块列表
            self.to_mine_blocks.append(new_block)
            # 清空self.transactions
            self.all_transactions = self.all_transactions[self.transaction_threshold:]
            self.last_update_time = cur_time

        # 判断是否达到挖矿时间阈值
        if cur_time - self.last_update_time >= self.mine_time_threshold:
            # 生成新的区块
            new_block = Block(cur_time, self.all_transactions, self.chain[-1].hash)
            # 将新的区块加入待挖矿区块列表
            self.to_mine_blocks.append(new_block)
            # 清空self.transactions
            self.all_transactions = []
            self.last_update_time = cur_time
        return self.to_mine_blocks