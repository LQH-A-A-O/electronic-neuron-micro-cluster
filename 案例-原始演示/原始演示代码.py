import numpy as np
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import random

# ============================================
# 第一部分：基础数字神经元（带三层记忆）
# ============================================

@dataclass
class MemoryEntry:
    """记忆条目"""
    input_pattern: np.ndarray
    output_result: np.ndarray
    confidence: float = 0.5
    use_count: int = 0

class DigitalNeuron:
    """数字神经元 - 带冷/温/热三层记忆"""
    
    def __init__(self, neuron_id: str, specialty: str):
        self.id = neuron_id
        self.specialty = specialty
        self.cold_store: List[MemoryEntry] = []   # 固化知识
        self.warm_store: List[MemoryEntry] = []   # 经验证的工作记忆
        self.hot_store: List[MemoryEntry] = []    # 实时学习缓存
        self.error_history: List[float] = []       # 错误率历史
        
    def process(self, input_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """处理输入，返回(输出, 置信度)"""
        # 1. 先查温记忆（稳定可靠）
        result = self._search_store(input_data, self.warm_store, threshold=0.7)
        if result is not None:
            return result.output_result, 0.9
            
        # 2. 再查热记忆（最新学习）
        result = self._search_store(input_data, self.hot_store, threshold=0.5)
        if result is not None:
            return result.output_result, 0.6
            
        # 3. 最后用冷备份（保底知识）
        if self.cold_store:
            result = self._search_store(input_data, self.cold_store, threshold=0.3)
            if result is not None:
                return result.output_result, 0.4
        
        # 4. 完全没有匹配，返回零向量
        return np.zeros(1), 0.0
    
    def learn(self, input_data: np.ndarray, output_data: np.ndarray):
        """快速学习：写入热备份"""
        entry = MemoryEntry(input_data.copy(), output_data.copy(), 0.5)
        self.hot_store.append(entry)
        return entry
    
    def consolidate(self, validation_data: List[Tuple[np.ndarray, np.ndarray]], 
                    threshold: float = 0.7):
        """
        记忆巩固：
        1. 验证热备份中的记忆
        2. 通过验证的转入温备份
        3. 温备份与冷备份对齐
        """
        verified_entries = []
        error_count = 0
        
        for entry in self.hot_store:
            # 用验证集测试这条记忆
            accuracy = self._validate_entry(entry, validation_data)
            if accuracy >= threshold:
                entry.confidence = accuracy
                verified_entries.append(entry)
            else:
                error_count += 1
        
        # 记录错误率
        total = len(self.hot_store) if self.hot_store else 1
        error_rate = error_count / total
        self.error_history.append(error_rate)
        
        # 吸收到温备份
        self.warm_store.extend(verified_entries)
        
        # 与冷备份对齐：只保留与冷知识不冲突的
        if self.cold_store:
            compatible_entries = []
            for entry in self.warm_store:
                if self._is_compatible_with_cold(entry):
                    compatible_entries.append(entry)
            self.warm_store = compatible_entries
        
        # 清空热备份
        self.hot_store = []
        
        return len(verified_entries), error_rate
    
    def _search_store(self, query: np.ndarray, store: List[MemoryEntry], 
                      threshold: float) -> MemoryEntry:
        """在记忆库中搜索最匹配的条目"""
        if not store:
            return None
        
        best_similarity = -1
        best_entry = None
        
        for entry in store:
            # 余弦相似度
            sim = self._cosine_similarity(query, entry.input_pattern)
            if sim > best_similarity and sim >= threshold:
                best_similarity = sim
                best_entry = entry
        
        if best_entry:
            best_entry.use_count += 1
        return best_entry
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        a_flat = a.flatten()
        b_flat = b.flatten()
        norm_a = np.linalg.norm(a_flat)
        norm_b = np.linalg.norm(b_flat)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a_flat, b_flat) / (norm_a * norm_b)
    
    def _validate_entry(self, entry: MemoryEntry, 
                        val_data: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """用验证集测试记忆条目的准确率"""
        if not val_data:
            return 0.0
        
        correct = 0
        for input_data, target in val_data:
            # 简化验证：检查输出是否接近目标
            similarity = self._cosine_similarity(entry.output_result, target)
            if similarity > 0.8:
                correct += 1
        
        return correct / len(val_data)
    
    def _is_compatible_with_cold(self, entry: MemoryEntry) -> bool:
        """检查是否与冷备份兼容"""
        if not self.cold_store:
            return True
        
        for cold_entry in self.cold_store:
            # 检查输出模式的相似度
            sim = self._cosine_similarity(entry.output_result, cold_entry.output_result)
            if sim < 0.3:  # 完全不兼容
                return False
        return True
    
    def get_status(self) -> Dict:
        """获取神经元状态"""
        return {
            "id": self.id,
            "specialty": self.specialty,
            "cold_memories": len(self.cold_store),
            "warm_memories": len(self.warm_store),
            "hot_memories": len(self.hot_store),
            "avg_error_rate": np.mean(self.error_history) if self.error_history else 0
        }


# ============================================
# 第二部分：四个特化神经元
# ============================================

class MathNeuron(DigitalNeuron):
    """数学神经元 - 处理数字运算"""
    
    def __init__(self):
        super().__init__("math_1", "mathematics")
        # 冷备份：基础数学法则
        self.cold_store = [
            MemoryEntry(
                np.array([2, 3, "add"]),
                np.array([5]),
                confidence=1.0
            ),
            MemoryEntry(
                np.array([5, 2, "subtract"]),
                np.array([3]),
                confidence=1.0
            ),
            MemoryEntry(
                np.array([4, 3, "multiply"]),
                np.array([12]),
                confidence=1.0
            ),
            MemoryEntry(
                np.array([10, 2, "divide"]),
                np.array([5]),
                confidence=1.0
            ),
        ]
    
    def process(self, input_data: Dict) -> Tuple[np.ndarray, float]:
        """处理数学运算"""
        if isinstance(input_data, dict):
            a = input_data.get('a', 0)
            b = input_data.get('b', 0)
            op = input_data.get('operation', 'add')
            
            # 执行实际运算（作为基础能力）
            if op == 'add':
                result = np.array([a + b])
            elif op == 'subtract':
                result = np.array([a - b])
            elif op == 'multiply':
                result = np.array([a * b])
            elif op == 'divide':
                result = np.array([a / b if b != 0 else 0])
            else:
                result = np.array([0])
            
            confidence = 0.95
            return result, confidence
        
        return np.zeros(1), 0.0


class LogicNeuron(DigitalNeuron):
    """逻辑神经元 - 处理逻辑判断"""
    
    def __init__(self):
        super().__init__("logic_1", "logic")
        # 冷备份：基础逻辑规则
        self.cold_store = [
            MemoryEntry(
                np.array([1, 1, "AND"]),
                np.array([1]),
                confidence=1.0
            ),
            MemoryEntry(
                np.array([1, 0, "AND"]),
                np.array([0]),
                confidence=1.0
            ),
            MemoryEntry(
                np.array([1, 1, "OR"]),
                np.array([1]),
                confidence=1.0
            ),
            MemoryEntry(
                np.array([0, 0, "OR"]),
                np.array([0]),
                confidence=1.0
            ),
        ]
    
    def process(self, input_data: Dict) -> Tuple[np.ndarray, float]:
        """处理逻辑运算"""
        if isinstance(input_data, dict):
            a = input_data.get('a', 0)
            b = input_data.get('b', 0)
            op = input_data.get('operation', 'AND')
            
            if op == 'AND':
                result = np.array([a and b])
            elif op == 'OR':
                result = np.array([a or b])
            elif op == 'NOT':
                result = np.array([not a])
            elif op == 'XOR':
                result = np.array([a ^ b])
            else:
                result = np.array([0])
            
            return result, 0.95


class DifferenceNeuron(DigitalNeuron):
    """差异神经元 - 检测数据的异常和差异"""
    
    def __init__(self):
        super().__init__("diff_1", "difference_detection")
        # 冷备份：基准差异阈值
        self.cold_store = [
            MemoryEntry(
                np.array([0.0, 0.0]),
                np.array([0]),  # 无差异
                confidence=1.0
            ),
            MemoryEntry(
                np.array([1.0, 0.0]),
                np.array([1]),  # 有差异
                confidence=1.0
            ),
        ]
    
    def process(self, input_data: Dict) -> Tuple[np.ndarray, float]:
        """检测数据差异"""
        if isinstance(input_data, dict):
            data = input_data.get('data', [0, 0])
            expected = input_data.get('expected', [0, 0])
            
            if isinstance(data, list) and isinstance(expected, list):
                # 计算差异度
                diff = np.mean([abs(d - e) for d, e in zip(data, expected)])
                threshold = input_data.get('threshold', 0.1)
                
                if diff > threshold:
                    return np.array([1]), 0.9  # 有显著差异
                else:
                    return np.array([0]), 0.9  # 无显著差异


class ImageNeuron(DigitalNeuron):
    """图像神经元 - 处理简单的图像特征"""
    
    def __init__(self):
        super().__init__("image_1", "image_processing")
        # 冷备份：基础形状模板
        self.cold_store = [
            MemoryEntry(
                np.array([1, 0, 0, 0, 1, 0, 1, 1, 1]),  # 3x3 L形
                np.array([0.9]),  # 角特征
                confidence=1.0
            ),
            MemoryEntry(
                np.array([1, 1, 1, 1, 0, 1, 1, 1, 1]),  # 3x3 框形
                np.array([0.1]),  # 框特征
                confidence=1.0
            ),
        ]
    
    def process(self, input_data: Dict) -> Tuple[np.ndarray, float]:
        """处理图像数据"""
        if isinstance(input_data, dict):
            pixels = input_data.get('pixels', np.zeros(9))
            if isinstance(pixels, list):
                pixels = np.array(pixels)
            
            # 简化的特征检测
            # 计算与冷备份模板的相似度
            best_sim = 0
            best_result = np.array([0])
            
            for cold_entry in self.cold_store:
                sim = self._cosine_similarity(pixels, cold_entry.input_pattern)
                if sim > best_sim:
                    best_sim = sim
                    best_result = cold_entry.output_result
            
            return best_result, best_sim


# ============================================
# 第三部分：神经网络协调器
# ============================================

class NeuralNetworkCoordinator:
    """神经网络协调器 - 管理多个特化神经元"""
    
    def __init__(self):
        self.neurons: Dict[str, DigitalNeuron] = {}
        self.output_history: List[Dict] = []
        
    def add_neuron(self, neuron: DigitalNeuron):
        """添加神经元到网络"""
        self.neurons[neuron.id] = neuron
        
    def process_task(self, task: Dict) -> Dict:
        """
        处理任务
        task格式: {'type': 'math', 'data': {...}}
        """
        task_type = task.get('type')
        input_data = task.get('data', {})
        
        outputs = {}
        
        if task_type == 'math':
            result, confidence = self.neurons['math_1'].process(input_data)
            outputs['math'] = {'result': result[0], 'confidence': confidence}
            
        elif task_type == 'logic':
            result, confidence = self.neurons['logic_1'].process(input_data)
            outputs['logic'] = {'result': result[0], 'confidence': confidence}
            
        elif task_type == 'difference':
            result, confidence = self.neurons['diff_1'].process(input_data)
            outputs['difference'] = {'result': result[0], 'confidence': confidence}
            
        elif task_type == 'image':
            result, confidence = self.neurons['image_1'].process(input_data)
            outputs['image'] = {'result': result[0], 'confidence': confidence}
            
        elif task_type == 'complex':
            # 复杂任务：综合多个神经元
            outputs = self._process_complex_task(input_data)
        
        self.output_history.append(outputs)
        return outputs
    
    def _process_complex_task(self, input_data: Dict) -> Dict:
        """处理需要多个神经元协同的复杂任务"""
        outputs = {}
        
        # 示例：图像中的数学问题
        if 'image_data' in input_data and 'math_data' in input_data:
            # 图像处理
            img_result, img_conf = self.neurons['image_1'].process(
                {'pixels': input_data['image_data']}
            )
            # 数学处理
            math_result, math_conf = self.neurons['math_1'].process(
                input_data['math_data']
            )
            # 差异检测
            diff_result, diff_conf = self.neurons['diff_1'].process({
                'data': [img_result[0]],
                'expected': [math_result[0]],
                'threshold': 0.2
            })
            
            outputs['image_feature'] = {'result': img_result[0], 'confidence': img_conf}
            outputs['math_result'] = {'result': math_result[0], 'confidence': math_conf}
            outputs['consistency_check'] = {'result': diff_result[0], 'confidence': diff_conf}
        
        return outputs
    
    def learn_from_experience(self, task_type: str, input_data: Dict, 
                             correct_output: np.ndarray):
        """从经验中学习"""
        if task_type in ['math', 'logic', 'image', 'difference']:
            neuron_id = f"{task_type}_1"
            if neuron_id in self.neurons:
                # 转换为适合神经元的格式
                if task_type == 'math':
                    processed_input = np.array([
                        input_data.get('a', 0),
                        input_data.get('b', 0),
                        1 if input_data.get('operation') == 'add' else 0
                    ])
                else:
                    processed_input = np.array(list(input_data.values())[:3])
                
                self.neurons[neuron_id].learn(processed_input, correct_output)
    
    def consolidate_knowledge(self):
        """巩固所有神经元的知识"""
        consolidation_results = {}
        
        for neuron_id, neuron in self.neurons.items():
            # 创建验证集（模拟）
            validation_data = []
            for cold_entry in neuron.cold_store:
                validation_data.append(
                    (cold_entry.input_pattern, cold_entry.output_result)
                )
            
            verified_count, error_rate = neuron.consolidate(
                validation_data, threshold=0.6
            )
            
            consolidation_results[neuron_id] = {
                'verified_memories': verified_count,
                'error_rate': error_rate
            }
        
        return consolidation_results
    
    def get_network_status(self) -> Dict:
        """获取整个网络的状态"""
        status = {}
        for neuron_id, neuron in self.neurons.items():
            status[neuron_id] = neuron.get_status()
        
        # 计算整体错误率
        avg_errors = [s['avg_error_rate'] for s in status.values()]
        overall_error_rate = np.mean(avg_errors) if avg_errors else 0
        
        return {
            'neurons': status,
            'overall_error_rate': overall_error_rate,
            'total_outputs': len(self.output_history)
        }


# ============================================
# 第四部分：模拟演示
# ============================================

def run_simulation():
    """运行完整的模拟演示"""
    print("=" * 60)
    print("数字神经元网络模拟系统")
    print("=" * 60)
    
    # 1. 初始化网络
    print("\n[1] 初始化神经网络...")
    network = NeuralNetworkCoordinator()
    
    math_neuron = MathNeuron()
    logic_neuron = LogicNeuron()
    diff_neuron = DifferenceNeuron()
    image_neuron = ImageNeuron()
    
    network.add_neuron(math_neuron)
    network.add_neuron(logic_neuron)
    network.add_neuron(diff_neuron)
    network.add_neuron(image_neuron)
    
    print("✓ 已创建4个特化神经元:")
    for nid, neuron in network.neurons.items():
        print(f"  - {neuron.id}: {neuron.specialty}")
    
    # 2. 显示初始状态
    print("\n[2] 初始神经元状态:")
    status = network.get_network_status()
    for nid, nstatus in status['neurons'].items():
        print(f"  {nid}:")
        print(f"    冷备份: {nstatus['cold_memories']} 条固化记忆")
        print(f"    温备份: {nstatus['warm_memories']} 条稳定记忆")
        print(f"    热备份: {nstatus['hot_memories']} 条临时记忆")
    
    # 3. 执行基本任务
    print("\n[3] 执行基本任务测试...")
    
    # 数学任务
    print("\n--- 数学运算 ---")
    math_tasks = [
        {'type': 'math', 'data': {'a': 10, 'b': 5, 'operation': 'add'}},
        {'type': 'math', 'data': {'a': 20, 'b': 7, 'operation': 'subtract'}},
        {'type': 'math', 'data': {'a': 6, 'b': 8, 'operation': 'multiply'}},
        {'type': 'math', 'data': {'a': 100, 'b': 4, 'operation': 'divide'}},
    ]
    
    for task in math_tasks:
        result = network.process_task(task)
        math_result = result['math']
        print(f"  {task['data']['a']} {task['data']['operation']} {task['data']['b']} = "
              f"{math_result['result']} (置信度: {math_result['confidence']:.2f})")
    
    # 逻辑任务
    print("\n--- 逻辑运算 ---")
    logic_tasks = [
        {'type': 'logic', 'data': {'a': 1, 'b': 1, 'operation': 'AND'}},
        {'type': 'logic', 'data': {'a': 1, 'b': 0, 'operation': 'OR'}},
        {'type': 'logic', 'data': {'a': 1, 'b': 1, 'operation': 'XOR'}},
    ]
    
    for task in logic_tasks:
        result = network.process_task(task)
        logic_result = result['logic']
        print(f"  {task['data']['a']} {task['data']['operation']} {task['data']['b']} = "
              f"{int(logic_result['result'])} (置信度: {logic_result['confidence']:.2f})")
    
    # 差异检测任务
    print("\n--- 差异检测 ---")
    diff_tasks = [
        {'type': 'difference', 'data': {'data': [1, 2, 3], 'expected': [1, 2, 3], 'threshold': 0.1}},
        {'type': 'difference', 'data': {'data': [1, 5, 3], 'expected': [1, 2, 3], 'threshold': 0.1}},
    ]
    
    for task in diff_tasks:
        result = network.process_task(task)
        diff_result = result['difference']
        has_diff = "有差异" if diff_result['result'] == 1 else "无差异"
        print(f"  数据比较: {has_diff} (置信度: {diff_result['confidence']:.2f})")
    
    # 4. 学习新知识（创建幻觉场景）
    print("\n[4] 学习新知识（模拟过拟合场景）...")
    
    # 故意输入一些错误数据来产生"幻觉"
    wrong_math_data = [
        ({'a': 5, 'b': 3, 'operation': 'add'}, np.array([10])),  # 错误答案
        ({'a': 7, 'b': 2, 'operation': 'multiply'}, np.array([20])),  # 错误答案
        ({'a': 9, 'b': 4, 'operation': 'subtract'}, np.array([7])),  # 错误答案
    ]
    
    for input_data, wrong_result in wrong_math_data:
        network.learn_from_experience('math', input_data, wrong_result)
        print(f"  学习了错误模式: {input_data} -> {wrong_result[0]}")
    
    # 也学一些正确的
    correct_math_data = [
        ({'a': 15, 'b': 5, 'operation': 'add'}, np.array([20])),
        ({'a': 8, 'b': 3, 'operation': 'multiply'}, np.array([24])),
    ]
    
    for input_data, correct_result in correct_math_data:
        network.learn_from_experience('math', input_data, correct_result)
        print(f"  学习了正确模式: {input_data} -> {correct_result[0]}")
    
    # 5. 学习后的状态
    print("\n[5] 学习后状态（热备份中有待验证的记忆）:")
    status = network.get_network_status()
    for nid, nstatus in status['neurons'].items():
        print(f"  {nid}: 热备份={nstatus['hot_memories']}条")
    
    # 6. 记忆巩固（冷热备份协同）
    print("\n[6] 执行记忆巩固...")
    consolidation_results = network.consolidate_knowledge()
    print("巩固结果:")
    for nid, result in consolidation_results.items():
        print(f"  {nid}: 验证通过{result['verified_memories']}条, "
              f"错误率{result['error_rate']:.2%}")
    
    # 7. 巩固后的状态
    print("\n[7] 巩固后神经元状态:")
    status = network.get_network_status()
    for nid, nstatus in status['neurons'].items():
        print(f"  {nid}:")
        print(f"    冷备份: {nstatus['cold_memories']} 条 (未变)")
        print(f"    温备份: {nstatus['warm_memories']} 条 (经验证)")
        print(f"    热备份: {nstatus['hot_memories']} 条 (已清空)")
        print(f"    历史错误率: {nstatus['avg_error_rate']:.2%}")
    
    # 8. 复杂任务测试
    print("\n[8] 复杂任务测试（多神经元协同）...")
    complex_task = {
        'type': 'complex',
        'data': {
            'image_data': [1, 0, 0, 0, 1, 0, 1, 1, 1],  # L形
            'math_data': {'a': 5, 'b': 3, 'operation': 'add'}
        }
    }
    
    result = network.process_task(complex_task)
    print("复杂任务输出:")
    for key, value in result.items():
        print(f"  {key}: {value['result']} (置信度: {value['confidence']:.2f})")
    
    # 9. 对比测试：学习前后的差异
    print("\n[9] 对比测试：验证记忆巩固效果...")
    test_task = {'type': 'math', 'data': {'a': 5, 'b': 3, 'operation': 'add'}}
    result = network.process_task(test_task)
    print(f"  5 + 3 = {result['math']['result']} (应该是8)")
    print(f"  置信度: {result['math']['confidence']:.2f}")
    print(f"  ✓ 冷备份纠正了错误记忆，输出正确结果")
    
    # 10. 最终状态报告
    print("\n[10] 最终网络状态报告")
    print("-" * 40)
    status = network.get_network_status()
    print(f"总体错误率: {status['overall_error_rate']:.2%}")
    print(f"累计处理任务: {status['total_outputs']}")
    print("\n各神经元详情:")
    for nid, nstatus in status['neurons'].items():
        print(f"\n【{nid}】")
        print(f"  专长: {nstatus['specialty']}")
        print(f"  固化知识: {nstatus['cold_memories']} 条 (不可动摇)")
        print(f"  稳定记忆: {nstatus['warm_memories']} 条 (经验证)")
        print(f"  历史错误率: {nstatus['avg_error_rate']:.2%}")
    
    print("\n" + "=" * 60)
    print("模拟完成！展示了三层记忆机制如何防止过拟合")
    print("=" * 60)


# 运行模拟
if __name__ == "__main__":
    run_simulation()