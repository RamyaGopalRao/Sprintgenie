# 🎯 Evaluation Systems in Python

## Complete Guide to Implementing Evaluation & Testing Frameworks

---

## 📚 Table of Contents

1. [Safe Expression Evaluation](#safe-expression-evaluation)
2. [Rule Engine Implementation](#rule-engine-implementation)
3. [AI/ML Model Evaluation](#aiml-model-evaluation)
4. [Business Rules Evaluation](#business-rules-evaluation)
5. [Code Evaluation Sandbox](#code-evaluation-sandbox)
6. [Performance Benchmarking](#performance-benchmarking)

---

## 1️⃣ Safe Expression Evaluation

### Understanding `eval()` and its Dangers

```python
"""
DANGEROUS - Never use in production:
"""
user_input = "os.system('rm -rf /')"  # Malicious code
result = eval(user_input)  # ❌ NEVER DO THIS!

"""
SAFER ALTERNATIVES:
1. ast.literal_eval() - For simple data structures
2. Custom expression parser
3. Sandboxed evaluation
4. Restricted namespace eval
"""

import ast
from typing import Any, Dict, Optional
import operator

# ================== SAFE LITERAL EVALUATION ==================

def safe_literal_eval(expression: str) -> Any:
    """
    Safely evaluate Python literal expressions.
    
    Only allows: strings, bytes, numbers, tuples, lists, dicts, sets, booleans, None
    
    Args:
        expression: String expression to evaluate
        
    Returns:
        Evaluated result
        
    Raises:
        ValueError: If expression contains unsafe operations
    """
    try:
        return ast.literal_eval(expression)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Invalid expression: {e}")

# Examples
safe_literal_eval("[1, 2, 3]")  # ✅ OK
safe_literal_eval("{'name': 'John', 'age': 30}")  # ✅ OK
safe_literal_eval("(1, 'hello', True)")  # ✅ OK
safe_literal_eval("os.system('ls')")  # ❌ Raises ValueError

# ================== CUSTOM EXPRESSION EVALUATOR ==================

class SafeExpressionEvaluator:
    """
    Safe expression evaluator with controlled operations.
    
    Allows only whitelisted operations and functions.
    """
    
    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: operator.and_,
        ast.Or: operator.or_,
        ast.Not: operator.not_,
    }
    
    # Allowed functions
    FUNCTIONS = {
        'abs': abs,
        'min': min,
        'max': max,
        'len': len,
        'sum': sum,
        'round': round,
        'int': int,
        'float': float,
        'str': str,
        'bool': bool,
    }
    
    def __init__(self, custom_vars: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator.
        
        Args:
            custom_vars: Custom variables to make available in expressions
        """
        self.variables = custom_vars or {}
    
    def evaluate(self, expression: str) -> Any:
        """
        Safely evaluate mathematical/logical expression.
        
        Examples:
            evaluate("2 + 3 * 4")  # 14
            evaluate("x > 10 and y < 20", {"x": 15, "y": 18})  # True
        
        Args:
            expression: Expression string
            
        Returns:
            Evaluation result
        """
        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")
    
    def _eval_node(self, node: ast.AST) -> Any:
        """Recursively evaluate AST node."""
        
        # Literal values (numbers, strings, etc.)
        if isinstance(node, ast.Constant):
            return node.value
        
        # Variables
        elif isinstance(node, ast.Name):
            if node.id not in self.variables:
                raise ValueError(f"Undefined variable: {node.id}")
            return self.variables[node.id]
        
        # Binary operations (2 + 3, x * y, etc.)
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op_func = self.OPERATORS.get(type(node.op))
            
            if not op_func:
                raise ValueError(f"Unsupported operator: {node.op}")
            
            return op_func(left, right)
        
        # Comparison operations (x > 10, y == 5, etc.)
        elif isinstance(node, ast.Compare):
            left = self._eval_node(node.left)
            
            for op, comparator in zip(node.ops, node.comparators):
                right = self._eval_node(comparator)
                op_func = self.OPERATORS.get(type(op))
                
                if not op_func:
                    raise ValueError(f"Unsupported comparison: {op}")
                
                if not op_func(left, right):
                    return False
                
                left = right
            
            return True
        
        # Boolean operations (x and y, a or b, not c)
        elif isinstance(node, ast.BoolOp):
            op_func = self.OPERATORS.get(type(node.op))
            values = [self._eval_node(v) for v in node.values]
            
            if isinstance(node.op, ast.And):
                return all(values)
            elif isinstance(node.op, ast.Or):
                return any(values)
        
        # Unary operations (not, -, +)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            
            if isinstance(node.op, ast.Not):
                return not operand
            elif isinstance(node.op, ast.USub):
                return -operand
            elif isinstance(node.op, ast.UAdd):
                return +operand
        
        # Function calls
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls allowed")
            
            func_name = node.func.id
            
            if func_name not in self.FUNCTIONS:
                raise ValueError(f"Function not allowed: {func_name}")
            
            func = self.FUNCTIONS[func_name]
            args = [self._eval_node(arg) for arg in node.args]
            
            return func(*args)
        
        # Lists, tuples
        elif isinstance(node, (ast.List, ast.Tuple)):
            return [self._eval_node(el) for el in node.elts]
        
        # Subscript (list[0], dict['key'])
        elif isinstance(node, ast.Subscript):
            value = self._eval_node(node.value)
            index = self._eval_node(node.slice)
            return value[index]
        
        else:
            raise ValueError(f"Unsupported operation: {type(node).__name__}")

# Usage Examples
evaluator = SafeExpressionEvaluator()

# Simple math
print(evaluator.evaluate("2 + 3 * 4"))  # 14
print(evaluator.evaluate("(10 + 5) / 3"))  # 5.0

# With variables
evaluator = SafeExpressionEvaluator({"x": 10, "y": 20, "z": 30})
print(evaluator.evaluate("x + y * z"))  # 610
print(evaluator.evaluate("x > 5 and y < 25"))  # True

# Functions
print(evaluator.evaluate("max(10, 20, 30)"))  # 30
print(evaluator.evaluate("abs(-42)"))  # 42
print(evaluator.evaluate("sum([1, 2, 3, 4])"))  # 10
```

---

## 2️⃣ Rule Engine Implementation

### Business Rules Evaluation System

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import re

class RuleOperator(Enum):
    """Supported rule operators."""
    EQUALS = "=="
    NOT_EQUALS = "!="
    GREATER_THAN = ">"
    GREATER_THAN_OR_EQUAL = ">="
    LESS_THAN = "<"
    LESS_THAN_OR_EQUAL = "<="
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IN = "in"
    NOT_IN = "not_in"
    MATCHES = "matches"  # Regex
    BETWEEN = "between"

@dataclass
class RuleCondition:
    """Single rule condition."""
    field: str
    operator: RuleOperator
    value: Any
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate condition against context.
        
        Args:
            context: Data to evaluate against
            
        Returns:
            True if condition passes
        """
        # Get field value from context
        field_value = self._get_field_value(context, self.field)
        
        # Apply operator
        if self.operator == RuleOperator.EQUALS:
            return field_value == self.value
        
        elif self.operator == RuleOperator.NOT_EQUALS:
            return field_value != self.value
        
        elif self.operator == RuleOperator.GREATER_THAN:
            return field_value > self.value
        
        elif self.operator == RuleOperator.GREATER_THAN_OR_EQUAL:
            return field_value >= self.value
        
        elif self.operator == RuleOperator.LESS_THAN:
            return field_value < self.value
        
        elif self.operator == RuleOperator.LESS_THAN_OR_EQUAL:
            return field_value <= self.value
        
        elif self.operator == RuleOperator.CONTAINS:
            return self.value in field_value
        
        elif self.operator == RuleOperator.NOT_CONTAINS:
            return self.value not in field_value
        
        elif self.operator == RuleOperator.IN:
            return field_value in self.value
        
        elif self.operator == RuleOperator.NOT_IN:
            return field_value not in self.value
        
        elif self.operator == RuleOperator.MATCHES:
            return bool(re.match(self.value, str(field_value)))
        
        elif self.operator == RuleOperator.BETWEEN:
            return self.value[0] <= field_value <= self.value[1]
        
        return False
    
    def _get_field_value(self, context: Dict[str, Any], field: str) -> Any:
        """
        Get field value from context, supporting nested fields.
        
        Examples:
            "user.age" -> context["user"]["age"]
            "items[0].price" -> context["items"][0]["price"]
        """
        parts = field.split('.')
        value = context
        
        for part in parts:
            # Handle array indexing
            if '[' in part:
                key, index = part.split('[')
                index = int(index.rstrip(']'))
                value = value[key][index]
            else:
                value = value[part]
        
        return value

@dataclass
class Rule:
    """
    Business rule with multiple conditions.
    
    Supports AND/OR logic.
    """
    rule_id: str
    name: str
    description: str
    conditions: List[RuleCondition]
    logic: str = "AND"  # "AND" or "OR"
    actions: List[Dict[str, Any]] = None
    priority: int = 0
    is_active: bool = True
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """
        Evaluate all conditions with specified logic.
        
        Args:
            context: Data to evaluate
            
        Returns:
            True if rule passes
        """
        if not self.is_active:
            return False
        
        results = [condition.evaluate(context) for condition in self.conditions]
        
        if self.logic == "AND":
            return all(results)
        elif self.logic == "OR":
            return any(results)
        else:
            raise ValueError(f"Invalid logic operator: {self.logic}")

class RuleEngine:
    """
    Rule engine for evaluating business rules.
    
    Features:
    - Priority-based rule execution
    - Action execution on rule match
    - Rule chaining
    - Audit logging
    """
    
    def __init__(self):
        self.rules: Dict[str, Rule] = {}
        self.actions: Dict[str, Callable] = {}
    
    def add_rule(self, rule: Rule) -> None:
        """Add rule to engine."""
        self.rules[rule.rule_id] = rule
    
    def register_action(self, action_name: str, action_func: Callable) -> None:
        """Register action that can be triggered by rules."""
        self.actions[action_name] = action_func
    
    async def evaluate(
        self,
        context: Dict[str, Any],
        rule_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate rules against context.
        
        Args:
            context: Data to evaluate
            rule_ids: Specific rules to evaluate (or all if None)
            
        Returns:
            Evaluation results with matched rules and executed actions
        """
        rules_to_check = (
            [self.rules[rid] for rid in rule_ids if rid in self.rules]
            if rule_ids
            else list(self.rules.values())
        )
        
        # Sort by priority
        rules_to_check.sort(key=lambda r: r.priority, reverse=True)
        
        matched_rules = []
        executed_actions = []
        
        for rule in rules_to_check:
            if rule.evaluate(context):
                matched_rules.append(rule.rule_id)
                
                # Execute actions
                if rule.actions:
                    for action_config in rule.actions:
                        action_name = action_config.get("action")
                        action_params = action_config.get("params", {})
                        
                        if action_name in self.actions:
                            result = await self._execute_action(
                                action_name,
                                context,
                                action_params
                            )
                            executed_actions.append({
                                "action": action_name,
                                "result": result
                            })
        
        return {
            "matched_rules": matched_rules,
            "executed_actions": executed_actions,
            "context": context
        }
    
    async def _execute_action(
        self,
        action_name: str,
        context: Dict[str, Any],
        params: Dict[str, Any]
    ) -> Any:
        """Execute action function."""
        action_func = self.actions[action_name]
        
        if asyncio.iscoroutinefunction(action_func):
            return await action_func(context, **params)
        else:
            return action_func(context, **params)

# ================== EXAMPLE: PRICING RULES ==================

# Define pricing rules
pricing_engine = RuleEngine()

# Rule 1: Enterprise discount
enterprise_discount_rule = Rule(
    rule_id="enterprise_discount",
    name="Enterprise Customer Discount",
    description="Apply 20% discount for enterprise customers",
    conditions=[
        RuleCondition("customer.type", RuleOperator.EQUALS, "enterprise"),
        RuleCondition("order.total", RuleOperator.GREATER_THAN, 10000)
    ],
    logic="AND",
    actions=[
        {
            "action": "apply_discount",
            "params": {"discount_percentage": 20}
        }
    ],
    priority=10
)

# Rule 2: Bulk order discount
bulk_discount_rule = Rule(
    rule_id="bulk_discount",
    name="Bulk Order Discount",
    description="Apply 10% discount for orders > 100 items",
    conditions=[
        RuleCondition("order.item_count", RuleOperator.GREATER_THAN, 100)
    ],
    actions=[
        {
            "action": "apply_discount",
            "params": {"discount_percentage": 10}
        }
    ],
    priority=5
)

# Rule 3: First-time customer
first_time_rule = Rule(
    rule_id="first_time_discount",
    name="First Time Customer",
    description="Apply 15% discount for first purchase",
    conditions=[
        RuleCondition("customer.order_count", RuleOperator.EQUALS, 0)
    ],
    actions=[
        {
            "action": "apply_discount",
            "params": {"discount_percentage": 15}
        },
        {
            "action": "send_welcome_email",
            "params": {}
        }
    ],
    priority=8
)

# Add rules to engine
pricing_engine.add_rule(enterprise_discount_rule)
pricing_engine.add_rule(bulk_discount_rule)
pricing_engine.add_rule(first_time_rule)

# Register actions
def apply_discount_action(context: Dict, discount_percentage: float) -> Dict:
    """Apply discount to order."""
    original_total = context["order"]["total"]
    discount_amount = original_total * (discount_percentage / 100)
    new_total = original_total - discount_amount
    
    context["order"]["total"] = new_total
    context["order"]["discount_applied"] = discount_percentage
    
    return {
        "discount_percentage": discount_percentage,
        "discount_amount": discount_amount,
        "new_total": new_total
    }

def send_welcome_email_action(context: Dict) -> Dict:
    """Send welcome email to new customer."""
    email = context["customer"]["email"]
    print(f"Sending welcome email to {email}")
    return {"email_sent": True}

pricing_engine.register_action("apply_discount", apply_discount_action)
pricing_engine.register_action("send_welcome_email", send_welcome_email_action)

# Evaluate rules
async def calculate_order_price():
    """Calculate order price with rules."""
    order_context = {
        "customer": {
            "type": "enterprise",
            "email": "customer@company.com",
            "order_count": 0
        },
        "order": {
            "total": 50000,
            "item_count": 150
        }
    }
    
    result = await pricing_engine.evaluate(order_context)
    
    print(f"Matched rules: {result['matched_rules']}")
    print(f"Final price: ₹{order_context['order']['total']}")
    # Output: Enterprise discount (20%) applied, then first-time (15%)

# asyncio.run(calculate_order_price())
```

---

## 3️⃣ AI/ML Model Evaluation

### Model Performance Evaluation Framework

```python
from typing import List, Dict, Tuple, Any, Callable
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    
    # Additional metrics
    mse: Optional[float] = None  # For regression
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    
    # NLP metrics
    bleu_score: Optional[float] = None  # For translation
    rouge_scores: Optional[Dict[str, float]] = None
    perplexity: Optional[float] = None
    
    # Performance metrics
    inference_time_ms: Optional[float] = None
    throughput_qps: Optional[float] = None
    
    # Business metrics
    user_satisfaction: Optional[float] = None
    cost_per_prediction: Optional[float] = None

class ModelEvaluator:
    """
    Comprehensive model evaluation framework.
    
    For Bhashini 2.0: Evaluate translation, STT, TTS models
    """
    
    def __init__(self):
        self.evaluation_history: List[Dict] = []
    
    def evaluate_classification(
        self,
        y_true: List[int],
        y_pred: List[int],
        labels: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate classification model.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label names
            
        Returns:
            Evaluation metrics
        """
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix
        )
        
        metrics = EvaluationMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, average='weighted', zero_division=0),
            recall=recall_score(y_true, y_pred, average='weighted', zero_division=0),
            f1_score=f1_score(y_true, y_pred, average='weighted', zero_division=0),
            confusion_matrix=confusion_matrix(y_true, y_pred).tolist()
        )
        
        return metrics
    
    def evaluate_translation(
        self,
        references: List[str],
        hypotheses: List[str],
        source_texts: Optional[List[str]] = None
    ) -> EvaluationMetrics:
        """
        Evaluate translation model.
        
        Uses BLEU, ROUGE, and other MT metrics.
        """
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge import Rouge
        
        # Calculate BLEU score
        bleu_scores = []
        smoothing = SmoothingFunction().method1
        
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = [ref.split()]  # BLEU expects list of references
            hyp_tokens = hyp.split()
            
            score = sentence_bleu(
                ref_tokens,
                hyp_tokens,
                smoothing_function=smoothing
            )
            bleu_scores.append(score)
        
        avg_bleu = np.mean(bleu_scores)
        
        # Calculate ROUGE scores
        rouge = Rouge()
        rouge_scores = rouge.get_scores(hypotheses, references, avg=True)
        
        return EvaluationMetrics(
            accuracy=0.0,  # Not applicable for translation
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=[],
            bleu_score=avg_bleu,
            rouge_scores={
                'rouge-1': rouge_scores['rouge-1']['f'],
                'rouge-2': rouge_scores['rouge-2']['f'],
                'rouge-l': rouge_scores['rouge-l']['f']
            }
        )
    
    def evaluate_speech_to_text(
        self,
        reference_texts: List[str],
        hypothesis_texts: List[str]
    ) -> EvaluationMetrics:
        """
        Evaluate STT model using WER (Word Error Rate).
        
        WER = (S + D + I) / N
        S = Substitutions, D = Deletions, I = Insertions, N = Total words
        """
        import jiwer
        
        # Calculate WER
        wer = jiwer.wer(reference_texts, hypothesis_texts)
        
        # Calculate CER (Character Error Rate)
        cer = jiwer.cer(reference_texts, hypothesis_texts)
        
        # Word accuracy
        accuracy = 1.0 - wer
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=0.0,
            recall=0.0,
            f1_score=0.0,
            confusion_matrix=[],
            bleu_score=1.0 - wer  # Inverse of WER as proxy
        )
    
    async def evaluate_model_performance(
        self,
        model_func: Callable,
        test_data: List[Dict],
        batch_size: int = 32
    ) -> Dict[str, Any]:
        """
        Evaluate model performance (latency, throughput).
        
        Args:
            model_func: Model inference function
            test_data: Test dataset
            batch_size: Batch size for inference
            
        Returns:
            Performance metrics
        """
        import time
        
        latencies = []
        total_start = time.time()
        
        # Process in batches
        for i in range(0, len(test_data), batch_size):
            batch = test_data[i:i+batch_size]
            
            batch_start = time.time()
            
            # Run inference
            if asyncio.iscoroutinefunction(model_func):
                results = await model_func(batch)
            else:
                results = model_func(batch)
            
            batch_time = time.time() - batch_start
            
            # Record per-item latency
            per_item_latency = (batch_time / len(batch)) * 1000  # ms
            latencies.extend([per_item_latency] * len(batch))
        
        total_time = time.time() - total_start
        
        return {
            "total_samples": len(test_data),
            "total_time_seconds": total_time,
            "throughput_qps": len(test_data) / total_time,
            "latency_p50_ms": np.percentile(latencies, 50),
            "latency_p95_ms": np.percentile(latencies, 95),
            "latency_p99_ms": np.percentile(latencies, 99),
            "avg_latency_ms": np.mean(latencies)
        }
    
    def save_evaluation_report(
        self,
        model_id: str,
        metrics: EvaluationMetrics,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Save evaluation report for tracking.
        
        Args:
            model_id: Model identifier
            metrics: Evaluation metrics
            metadata: Additional metadata
            
        Returns:
            Report ID
        """
        import uuid
        
        report = {
            "report_id": str(uuid.uuid4()),
            "model_id": model_id,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics.__dict__,
            "metadata": metadata
        }
        
        self.evaluation_history.append(report)
        
        # Save to database/file
        with open(f"evaluations/{report['report_id']}.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        return report["report_id"]

# ================== EXAMPLE: TRANSLATION MODEL EVALUATION ==================

async def evaluate_translation_model():
    """Evaluate Bhashini translation model."""
    
    evaluator = ModelEvaluator()
    
    # Test dataset
    test_data = [
        {
            "source": "नमस्ते, आप कैसे हैं?",  # Hindi
            "reference": "Hello, how are you?",  # English
            "source_lang": "hi",
            "target_lang": "en"
        },
        # ... more test cases
    ]
    
    # Run model inference
    hypotheses = []
    references = []
    
    for item in test_data:
        # Call translation model
        result = await translation_model.translate(
            text=item["source"],
            source_lang=item["source_lang"],
            target_lang=item["target_lang"]
        )
        
        hypotheses.append(result["translated_text"])
        references.append(item["reference"])
    
    # Evaluate translation quality
    quality_metrics = evaluator.evaluate_translation(
        references=references,
        hypotheses=hypotheses
    )
    
    print(f"BLEU Score: {quality_metrics.bleu_score:.4f}")
    print(f"ROUGE-L: {quality_metrics.rouge_scores['rouge-l']:.4f}")
    
    # Evaluate performance
    performance_metrics = await evaluator.evaluate_model_performance(
        model_func=translation_model.translate_batch,
        test_data=test_data,
        batch_size=32
    )
    
    print(f"Throughput: {performance_metrics['throughput_qps']:.2f} QPS")
    print(f"P95 Latency: {performance_metrics['latency_p95_ms']:.2f} ms")
    
    # Save evaluation report
    report_id = evaluator.save_evaluation_report(
        model_id="hindi-english-v2",
        metrics=quality_metrics,
        metadata={
            "test_dataset_size": len(test_data),
            "performance": performance_metrics,
            "model_version": "2.0",
            "evaluated_by": "test_suite"
        }
    )
    
    return report_id
```

---

## 4️⃣ Business Rules Evaluation

### Dynamic Business Rules System

```python
from typing import Protocol, runtime_checkable
from decimal import Decimal

@runtime_checkable
class RuleContext(Protocol):
    """Protocol for rule evaluation context."""
    
    def get_value(self, path: str) -> Any:
        """Get value from context by path."""
        ...

class DynamicRuleEvaluator:
    """
    Evaluates business rules defined in JSON/YAML.
    
    Use case: Configure rules without code deployment.
    """
    
    def __init__(self):
        self.custom_functions: Dict[str, Callable] = {}
    
    def register_function(self, name: str, func: Callable) -> None:
        """Register custom function for use in rules."""
        self.custom_functions[name] = func
    
    def evaluate_rule_from_config(
        self,
        rule_config: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate rule from configuration.
        
        Example rule_config:
        {
            "conditions": [
                {"field": "amount", "operator": ">", "value": 1000},
                {"field": "currency", "operator": "==", "value": "INR"}
            ],
            "logic": "AND"
        }
        """
        conditions = rule_config.get("conditions", [])
        logic = rule_config.get("logic", "AND")
        
        results = []
        
        for condition in conditions:
            field = condition["field"]
            operator = condition["operator"]
            expected_value = condition["value"]
            
            # Get actual value from context
            actual_value = self._get_nested_value(context, field)
            
            # Evaluate condition
            result = self._evaluate_condition(actual_value, operator, expected_value)
            results.append(result)
        
        # Apply logic
        if logic == "AND":
            return all(results)
        elif logic == "OR":
            return any(results)
        else:
            return False
    
    def _get_nested_value(self, context: Dict, path: str) -> Any:
        """Get nested value from context using dot notation."""
        keys = path.split('.')
        value = context
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, (list, tuple)) and key.isdigit():
                value = value[int(key)]
            else:
                return None
        
        return value
    
    def _evaluate_condition(
        self,
        actual: Any,
        operator: str,
        expected: Any
    ) -> bool:
        """Evaluate single condition."""
        ops = {
            '==': lambda a, e: a == e,
            '!=': lambda a, e: a != e,
            '>': lambda a, e: a > e,
            '>=': lambda a, e: a >= e,
            '<': lambda a, e: a < e,
            '<=': lambda a, e: a <= e,
            'in': lambda a, e: a in e,
            'not in': lambda a, e: a not in e,
            'contains': lambda a, e: e in a if isinstance(a, (str, list)) else False,
            'startswith': lambda a, e: a.startswith(e) if isinstance(a, str) else False,
            'endswith': lambda a, e: a.endswith(e) if isinstance(a, str) else False,
        }
        
        op_func = ops.get(operator)
        if not op_func:
            raise ValueError(f"Unknown operator: {operator}")
        
        return op_func(actual, expected)

# ================== EXAMPLE: SUBSCRIPTION ELIGIBILITY ==================

rule_evaluator = DynamicRuleEvaluator()

# Load rules from JSON configuration
subscription_rules = {
    "premium_eligibility": {
        "conditions": [
            {"field": "user.account_age_days", "operator": ">=", "value": 30},
            {"field": "user.verified_email", "operator": "==", "value": True},
            {"field": "user.payment_method", "operator": "!=", "value": None}
        ],
        "logic": "AND"
    },
    "free_tier_limit": {
        "conditions": [
            {"field": "subscription.tier", "operator": "==", "value": "free"},
            {"field": "usage.requests_this_month", "operator": ">=", "value": 1000}
        ],
        "logic": "AND"
    }
}

# Evaluate eligibility
user_context = {
    "user": {
        "account_age_days": 45,
        "verified_email": True,
        "payment_method": "credit_card"
    },
    "subscription": {
        "tier": "free"
    },
    "usage": {
        "requests_this_month": 1200
    }
}

# Check premium eligibility
is_eligible = rule_evaluator.evaluate_rule_from_config(
    subscription_rules["premium_eligibility"],
    user_context
)
print(f"Premium eligible: {is_eligible}")  # True

# Check if free tier limit exceeded
limit_exceeded = rule_evaluator.evaluate_rule_from_config(
    subscription_rules["free_tier_limit"],
    user_context
)
print(f"Free tier limit exceeded: {limit_exceeded}")  # True
```

---

## 5️⃣ Code Evaluation Sandbox

### Safe Code Execution Environment

```python
import sys
import io
import ast
from contextlib import redirect_stdout, redirect_stderr
from typing import Dict, Any, Optional
import multiprocessing
import signal
from dataclasses import dataclass

@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool
    output: str
    error: Optional[str] = None
    execution_time_ms: float = 0
    memory_used_mb: float = 0

class SafeCodeExecutor:
    """
    Execute Python code in restricted environment.
    
    Use case: Online code evaluation, testing user-submitted code
    
    Security measures:
    - Restricted builtins
    - Import restrictions
    - Timeout enforcement
    - Memory limits
    - No file system access
    """
    
    # Allowed built-in functions
    SAFE_BUILTINS = {
        'abs': abs,
        'all': all,
        'any': any,
        'bin': bin,
        'bool': bool,
        'chr': chr,
        'dict': dict,
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'hex': hex,
        'int': int,
        'isinstance': isinstance,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'range': range,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'type': type,
        'zip': zip,
        'print': print,  # Captured to output
    }
    
    def __init__(self, timeout_seconds: int = 5, memory_limit_mb: int = 256):
        self.timeout_seconds = timeout_seconds
        self.memory_limit_mb = memory_limit_mb
    
    def validate_code(self, code: str) -> bool:
        """
        Validate code for dangerous operations.
        
        Checks:
        - No imports of restricted modules
        - No dangerous built-ins (eval, exec, compile)
        - No file operations
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return False
        
        # Check for imports
        for node in ast.walk(tree):
            # Block all imports
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return False
            
            # Block dangerous function calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['eval', 'exec', 'compile', '__import__', 'open']:
                        return False
        
        return True
    
    def execute(self, code: str, test_cases: Optional[List[Dict]] = None) -> ExecutionResult:
        """
        Execute code in sandboxed environment.
        
        Args:
            code: Python code to execute
            test_cases: Optional test cases to run
            
        Returns:
            Execution result
        """
        import time
        import resource
        
        # Validate code first
        if not self.validate_code(code):
            return ExecutionResult(
                success=False,
                output="",
                error="Code contains restricted operations"
            )
        
        # Capture output
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        
        start_time = time.time()
        
        try:
            # Set memory limit (Unix only)
            try:
                resource.setrlimit(
                    resource.RLIMIT_AS,
                    (self.memory_limit_mb * 1024 * 1024, -1)
                )
            except:
                pass  # Windows doesn't support resource limits
            
            # Create restricted globals
            safe_globals = {
                '__builtins__': self.SAFE_BUILTINS,
                '__name__': '__main__',
                '__doc__': None
            }
            
            # Execute code with timeout
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Use alarm for timeout (Unix only)
                def timeout_handler(signum, frame):
                    raise TimeoutError("Execution timed out")
                
                try:
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(self.timeout_seconds)
                except:
                    pass  # Windows doesn't support signals
                
                # Compile and execute
                compiled_code = compile(code, '<string>', 'exec')
                exec(compiled_code, safe_globals)
                
                # Cancel alarm
                try:
                    signal.alarm(0)
                except:
                    pass
                
                # Run test cases if provided
                if test_cases:
                    self._run_test_cases(test_cases, safe_globals, stdout_capture)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return ExecutionResult(
                success=True,
                output=stdout_capture.getvalue(),
                execution_time_ms=execution_time_ms
            )
            
        except TimeoutError:
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"Execution timeout ({self.timeout_seconds}s exceeded)"
            )
        
        except MemoryError:
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=f"Memory limit exceeded ({self.memory_limit_mb}MB)"
            )
        
        except Exception as e:
            return ExecutionResult(
                success=False,
                output=stdout_capture.getvalue(),
                error=str(e)
            )
    
    def _run_test_cases(
        self,
        test_cases: List[Dict],
        globals_dict: Dict,
        output_stream: io.StringIO
    ):
        """Run test cases against executed code."""
        for i, test_case in enumerate(test_cases):
            func_name = test_case.get("function")
            inputs = test_case.get("inputs", [])
            expected = test_case.get("expected")
            
            if func_name not in globals_dict:
                output_stream.write(f"\nTest {i+1} FAILED: Function '{func_name}' not found\n")
                continue
            
            func = globals_dict[func_name]
            
            try:
                result = func(*inputs)
                
                if result == expected:
                    output_stream.write(f"\nTest {i+1} PASSED ✓\n")
                else:
                    output_stream.write(
                        f"\nTest {i+1} FAILED ✗\n"
                        f"  Expected: {expected}\n"
                        f"  Got: {result}\n"
                    )
            except Exception as e:
                output_stream.write(f"\nTest {i+1} ERROR: {str(e)}\n")

# ================== EXAMPLE USAGE ==================

executor = SafeCodeExecutor(timeout_seconds=5, memory_limit_mb=128)

# User-submitted code
user_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print("Fibonacci of 10:", fibonacci(10))
"""

# Test cases
test_cases = [
    {"function": "fibonacci", "inputs": [0], "expected": 0},
    {"function": "fibonacci", "inputs": [1], "expected": 1},
    {"function": "fibonacci", "inputs": [5], "expected": 5},
    {"function": "fibonacci", "inputs": [10], "expected": 55},
]

# Execute
result = executor.execute(user_code, test_cases)

if result.success:
    print("✓ Code executed successfully")
    print(f"Output:\n{result.output}")
    print(f"Execution time: {result.execution_time_ms:.2f}ms")
else:
    print("✗ Execution failed")
    print(f"Error: {result.error}")
```

---

## 6️⃣ Performance Benchmarking

### Comprehensive Benchmarking System

```python
import time
import statistics
from typing import Callable, List, Dict, Any
from dataclasses import dataclass
import psutil
import tracemalloc

@dataclass
class BenchmarkResult:
    """Result of performance benchmark."""
    function_name: str
    iterations: int
    
    # Time metrics
    total_time_seconds: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    
    # Memory metrics
    peak_memory_mb: float
    memory_increase_mb: float
    
    # Throughput
    operations_per_second: float

class PerformanceBenchmark:
    """
    Benchmark Python code performance.
    
    Measures:
    - Execution time
    - Memory usage
    - Throughput
    """
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def benchmark(
        self,
        func: Callable,
        iterations: int = 1000,
        warmup: int = 10,
        *args,
        **kwargs
    ) -> BenchmarkResult:
        """
        Benchmark function performance.
        
        Args:
            func: Function to benchmark
            iterations: Number of iterations
            warmup: Warmup iterations (excluded from timing)
            
        Returns:
            Benchmark results
        """
        # Warmup
        for _ in range(warmup):
            func(*args, **kwargs)
        
        # Start memory tracking
        tracemalloc.start()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Benchmark
        times_ms = []
        
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            end = time.perf_counter()
            
            times_ms.append((end - start) * 1000)
        
        # Memory metrics
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate statistics
        total_time = sum(times_ms) / 1000
        
        result = BenchmarkResult(
            function_name=func.__name__,
            iterations=iterations,
            total_time_seconds=total_time,
            avg_time_ms=statistics.mean(times_ms),
            min_time_ms=min(times_ms),
            max_time_ms=max(times_ms),
            median_time_ms=statistics.median(times_ms),
            std_dev_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
            peak_memory_mb=peak / 1024 / 1024,
            memory_increase_mb=end_memory - start_memory,
            operations_per_second=iterations / total_time
        )
        
        self.results.append(result)
        return result
    
    def compare(
        self,
        functions: List[Tuple[str, Callable]],
        iterations: int = 1000
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare performance of multiple implementations.
        
        Args:
            functions: List of (name, function) tuples
            iterations: Iterations per function
            
        Returns:
            Comparison results
        """
        results = {}
        
        for name, func in functions:
            result = self.benchmark(func, iterations=iterations)
            results[name] = result
        
        # Print comparison
        print("\n" + "="*80)
        print(f"Performance Comparison ({iterations} iterations)")
        print("="*80)
        print(f"{'Function':<30} {'Avg Time':<15} {'Throughput':<20} {'Memory':<15}")
        print("-"*80)
        
        for name, result in results.items():
            print(
                f"{name:<30} "
                f"{result.avg_time_ms:>10.4f} ms  "
                f"{result.operations_per_second:>15.2f} ops/s  "
                f"{result.peak_memory_mb:>10.2f} MB"
            )
        
        return results

# ================== EXAMPLE: COMPARE IMPLEMENTATIONS ==================

benchmark = PerformanceBenchmark()

# Implementation 1: List comprehension
def sum_squares_list_comp(n: int) -> int:
    return sum([x**2 for x in range(n)])

# Implementation 2: Generator expression
def sum_squares_generator(n: int) -> int:
    return sum(x**2 for x in range(n))

# Implementation 3: NumPy
def sum_squares_numpy(n: int) -> int:
    import numpy as np
    arr = np.arange(n)
    return np.sum(arr**2)

# Compare implementations
results = benchmark.compare([
    ("List Comprehension", lambda: sum_squares_list_comp(10000)),
    ("Generator Expression", lambda: sum_squares_generator(10000)),
    ("NumPy", lambda: sum_squares_numpy(10000))
])

# Output:
# Performance Comparison (1000 iterations)
# Function                       Avg Time        Throughput           Memory
# List Comprehension                2.3451 ms       426.42 ops/s       0.15 MB
# Generator Expression              2.1203 ms       471.63 ops/s       0.08 MB
# NumPy                             0.8912 ms      1122.01 ops/s       0.42 MB
```

---

## 7️⃣ A/B Testing Evaluation

### A/B Test Statistical Evaluation

```python
from typing import List, Tuple
import numpy as np
from scipy import stats
from dataclasses import dataclass

@dataclass
class ABTestResult:
    """A/B test statistical results."""
    variant_a_mean: float
    variant_b_mean: float
    improvement_percentage: float
    p_value: float
    is_significant: bool
    confidence_level: float
    sample_size_a: int
    sample_size_b: int
    recommendation: str

class ABTestEvaluator:
    """
    Statistical evaluation for A/B tests.
    
    Use case: Evaluate different model versions, UI changes, algorithms
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize A/B test evaluator.
        
        Args:
            significance_level: p-value threshold (default: 0.05 = 95% confidence)
        """
        self.significance_level = significance_level
    
    def evaluate_means(
        self,
        variant_a_data: List[float],
        variant_b_data: List[float],
        metric_name: str = "conversion_rate"
    ) -> ABTestResult:
        """
        Evaluate A/B test for continuous metrics.
        
        Uses t-test for statistical significance.
        
        Args:
            variant_a_data: Control group measurements
            variant_b_data: Treatment group measurements
            metric_name: Name of metric being tested
            
        Returns:
            Statistical evaluation results
        """
        # Calculate statistics
        mean_a = np.mean(variant_a_data)
        mean_b = np.mean(variant_b_data)
        
        # Perform t-test
        t_statistic, p_value = stats.ttest_ind(variant_a_data, variant_b_data)
        
        # Calculate improvement
        improvement_pct = ((mean_b - mean_a) / mean_a) * 100
        
        # Determine significance
        is_significant = p_value < self.significance_level
        
        # Make recommendation
        if is_significant:
            if improvement_pct > 0:
                recommendation = f"Deploy Variant B - {improvement_pct:.2f}% improvement (statistically significant)"
            else:
                recommendation = f"Keep Variant A - Variant B performs {abs(improvement_pct):.2f}% worse"
        else:
            recommendation = "Insufficient evidence - continue testing or increase sample size"
        
        return ABTestResult(
            variant_a_mean=mean_a,
            variant_b_mean=mean_b,
            improvement_percentage=improvement_pct,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=(1 - self.significance_level) * 100,
            sample_size_a=len(variant_a_data),
            sample_size_b=len(variant_b_data),
            recommendation=recommendation
        )
    
    def evaluate_proportions(
        self,
        conversions_a: int,
        visitors_a: int,
        conversions_b: int,
        visitors_b: int
    ) -> ABTestResult:
        """
        Evaluate A/B test for conversion rates.
        
        Uses z-test for proportions.
        """
        # Calculate conversion rates
        rate_a = conversions_a / visitors_a
        rate_b = conversions_b / visitors_b
        
        # Calculate pooled proportion
        pooled_proportion = (conversions_a + conversions_b) / (visitors_a + visitors_b)
        
        # Calculate standard error
        se = np.sqrt(
            pooled_proportion * (1 - pooled_proportion) * (1/visitors_a + 1/visitors_b)
        )
        
        # Calculate z-statistic
        z_stat = (rate_b - rate_a) / se
        
        # Calculate p-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Improvement percentage
        improvement_pct = ((rate_b - rate_a) / rate_a) * 100
        
        is_significant = p_value < self.significance_level
        
        if is_significant:
            if improvement_pct > 0:
                recommendation = f"Deploy Variant B - {improvement_pct:.2f}% lift in conversion rate"
            else:
                recommendation = f"Keep Variant A - {abs(improvement_pct):.2f}% drop in conversion"
        else:
            recommendation = f"Not statistically significant (p={p_value:.4f}). Need more data."
        
        return ABTestResult(
            variant_a_mean=rate_a,
            variant_b_mean=rate_b,
            improvement_percentage=improvement_pct,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=(1 - self.significance_level) * 100,
            sample_size_a=visitors_a,
            sample_size_b=visitors_b,
            recommendation=recommendation
        )
    
    def calculate_required_sample_size(
        self,
        baseline_rate: float,
        minimum_detectable_effect: float,
        power: float = 0.8
    ) -> int:
        """
        Calculate required sample size for A/B test.
        
        Args:
            baseline_rate: Current conversion rate (e.g., 0.05 = 5%)
            minimum_detectable_effect: Minimum improvement to detect (e.g., 0.1 = 10% relative improvement)
            power: Statistical power (default: 0.8 = 80%)
            
        Returns:
            Required sample size per variant
        """
        from statsmodels.stats.power import zt_ind_solve_power
        
        # Effect size (Cohen's h for proportions)
        p1 = baseline_rate
        p2 = baseline_rate * (1 + minimum_detectable_effect)
        
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Calculate sample size
        sample_size = zt_ind_solve_power(
            effect_size=effect_size,
            alpha=self.significance_level,
            power=power,
            alternative='two-sided'
        )
        
        return int(np.ceil(sample_size))

# ================== EXAMPLE: MODEL COMPARISON ==================

# Compare two translation models
evaluator = ABTestEvaluator(significance_level=0.05)

# Model A (current)
model_a_bleu_scores = [0.42, 0.45, 0.41, 0.43, 0.44, 0.42, 0.45, 0.43, 0.41, 0.44]  # 10 samples

# Model B (new)
model_b_bleu_scores = [0.48, 0.50, 0.47, 0.49, 0.51, 0.48, 0.50, 0.49, 0.47, 0.50]  # 10 samples

result = evaluator.evaluate_means(
    variant_a_data=model_a_bleu_scores,
    variant_b_data=model_b_bleu_scores,
    metric_name="BLEU Score"
)

print(f"Model A average BLEU: {result.variant_a_mean:.4f}")
print(f"Model B average BLEU: {result.variant_b_mean:.4f}")
print(f"Improvement: {result.improvement_percentage:.2f}%")
print(f"P-value: {result.p_value:.4f}")
print(f"Significant: {result.is_significant}")
print(f"Recommendation: {result.recommendation}")

# Calculate sample size needed
required_samples = evaluator.calculate_required_sample_size(
    baseline_rate=0.43,  # Current BLEU score
    minimum_detectable_effect=0.10,  # Want to detect 10% improvement
    power=0.8
)
print(f"\nRequired sample size per variant: {required_samples}")
```

---

## 8️⃣ Evaluation for Bhashini Platform

### Translation Quality Evaluation

```python
class TranslationEvaluator:
    """
    Evaluate translation quality for Bhashini models.
    
    Metrics:
    - BLEU (Bilingual Evaluation Understudy)
    - ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
    - METEOR (Metric for Evaluation of Translation with Explicit ORdering)
    - TER (Translation Edit Rate)
    - Human evaluation scores
    """
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_translation_batch(
        self,
        source_texts: List[str],
        reference_translations: List[str],
        hypothesis_translations: List[str],
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Comprehensive translation evaluation.
        
        Args:
            source_texts: Original texts
            reference_translations: Ground truth translations
            hypothesis_translations: Model-generated translations
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Complete evaluation metrics
        """
        from sacrebleu import corpus_bleu
        from rouge import Rouge
        
        # BLEU Score
        bleu = corpus_bleu(
            hypothesis_translations,
            [reference_translations]
        )
        
        # ROUGE Scores
        rouge = Rouge()
        rouge_scores = rouge.get_scores(
            hypothesis_translations,
            reference_translations,
            avg=True
        )
        
        # Character-level accuracy (for Indic languages)
        char_accuracy = self._calculate_character_accuracy(
            reference_translations,
            hypothesis_translations
        )
        
        # Word-level accuracy
        word_accuracy = self._calculate_word_accuracy(
            reference_translations,
            hypothesis_translations
        )
        
        # Semantic similarity (using embeddings)
        semantic_similarity = await self._calculate_semantic_similarity(
            reference_translations,
            hypothesis_translations,
            target_lang
        )
        
        results = {
            "language_pair": f"{source_lang}-{target_lang}",
            "sample_count": len(source_texts),
            "bleu_score": bleu.score,
            "rouge_scores": {
                "rouge-1": rouge_scores["rouge-1"]["f"],
                "rouge-2": rouge_scores["rouge-2"]["f"],
                "rouge-l": rouge_scores["rouge-l"]["f"]
            },
            "character_accuracy": char_accuracy,
            "word_accuracy": word_accuracy,
            "semantic_similarity": semantic_similarity,
            "timestamp": datetime.now().isoformat()
        }
        
        self.evaluation_history.append(results)
        return results
    
    def _calculate_character_accuracy(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """
        Calculate character-level accuracy.
        
        Important for Indic languages with complex scripts.
        """
        total_chars = 0
        correct_chars = 0
        
        for ref, hyp in zip(references, hypotheses):
            total_chars += len(ref)
            
            # Count matching characters
            min_len = min(len(ref), len(hyp))
            correct_chars += sum(1 for i in range(min_len) if ref[i] == hyp[i])
        
        return correct_chars / total_chars if total_chars > 0 else 0.0
    
    def _calculate_word_accuracy(
        self,
        references: List[str],
        hypotheses: List[str]
    ) -> float:
        """Calculate word-level accuracy."""
        from difflib import SequenceMatcher
        
        accuracies = []
        
        for ref, hyp in zip(references, hypotheses):
            ref_words = ref.split()
            hyp_words = hyp.split()
            
            matcher = SequenceMatcher(None, ref_words, hyp_words)
            accuracy = matcher.ratio()
            accuracies.append(accuracy)
        
        return np.mean(accuracies)
    
    async def _calculate_semantic_similarity(
        self,
        references: List[str],
        hypotheses: List[str],
        language: str
    ) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Uses multilingual models like mBERT or XLM-RoBERTa.
        """
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Load multilingual model
        model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # Get embeddings
        ref_embeddings = model.encode(references)
        hyp_embeddings = model.encode(hypotheses)
        
        # Calculate cosine similarity
        similarities = [
            cosine_similarity([ref_emb], [hyp_emb])[0][0]
            for ref_emb, hyp_emb in zip(ref_embeddings, hyp_embeddings)
        ]
        
        return np.mean(similarities)
    
    def evaluate_with_human_feedback(
        self,
        translation_pairs: List[Tuple[str, str]],
        human_ratings: List[int]  # 1-5 scale
    ) -> Dict[str, Any]:
        """
        Incorporate human evaluation.
        
        Args:
            translation_pairs: (reference, hypothesis) pairs
            human_ratings: Human quality ratings (1-5)
            
        Returns:
            Human evaluation metrics
        """
        return {
            "avg_human_rating": np.mean(human_ratings),
            "median_rating": np.median(human_ratings),
            "rating_distribution": {
                rating: human_ratings.count(rating)
                for rating in range(1, 6)
            },
            "percentage_good_or_better": (
                sum(1 for r in human_ratings if r >= 4) / len(human_ratings) * 100
            )
        }

# ================== EXAMPLE: EVALUATE HINDI-ENGLISH TRANSLATION ==================

async def evaluate_bhashini_translation_model():
    """Evaluate Bhashini translation model."""
    
    evaluator = TranslationEvaluator()
    
    # Test dataset
    test_data = [
        {
            "source": "नमस्ते, आप कैसे हैं?",
            "reference": "Hello, how are you?",
            "hypothesis": "Hello, how are you?",
            "source_lang": "hi",
            "target_lang": "en"
        },
        {
            "source": "मैं ठीक हूँ, धन्यवाद।",
            "reference": "I am fine, thank you.",
            "hypothesis": "I am good, thanks.",
            "source_lang": "hi",
            "target_lang": "en"
        },
        # ... more test cases
    ]
    
    sources = [item["source"] for item in test_data]
    references = [item["reference"] for item in test_data]
    hypotheses = [item["hypothesis"] for item in test_data]
    
    # Evaluate
    results = await evaluator.evaluate_translation_batch(
        source_texts=sources,
        reference_translations=references,
        hypothesis_translations=hypotheses,
        source_lang="hi",
        target_lang="en"
    )
    
    print(f"Translation Quality Evaluation (Hindi → English)")
    print(f"BLEU Score: {results['bleu_score']:.4f}")
    print(f"ROUGE-L: {results['rouge_scores']['rouge-l']:.4f}")
    print(f"Character Accuracy: {results['character_accuracy']:.4f}")
    print(f"Semantic Similarity: {results['semantic_similarity']:.4f}")
    
    # Human evaluation
    human_ratings = [4, 5, 4, 4, 3, 5, 4, 4, 5, 4]  # Ratings from evaluators
    
    human_results = evaluator.evaluate_with_human_feedback(
        translation_pairs=list(zip(references, hypotheses)),
        human_ratings=human_ratings
    )
    
    print(f"\nHuman Evaluation:")
    print(f"Average Rating: {human_results['avg_human_rating']:.2f}/5")
    print(f"Good or Better: {human_results['percentage_good_or_better']:.1f}%")

# asyncio.run(evaluate_bhashini_translation_model())
```

---

## 9️⃣ FastAPI Integration

### Evaluation API Service

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import asyncio

app = FastAPI(title="Evaluation Service")

# Request/Response models
class EvaluationRequest(BaseModel):
    """Request to evaluate code or expressions."""
    code: Optional[str] = None
    expression: Optional[str] = None
    test_cases: Optional[List[Dict]] = None
    variables: Optional[Dict[str, Any]] = None

class EvaluationResponse(BaseModel):
    """Evaluation result response."""
    success: bool
    result: Optional[Any] = None
    output: Optional[str] = None
    error: Optional[str] = None
    execution_time_ms: float
    evaluation_id: str

class ModelEvaluationRequest(BaseModel):
    """Request to evaluate model."""
    model_id: str
    test_dataset: List[Dict]
    evaluation_type: str  # "translation", "stt", "classification"

# Endpoints
@app.post("/evaluate/expression", response_model=EvaluationResponse)
async def evaluate_expression(request: EvaluationRequest):
    """
    Safely evaluate mathematical/logical expression.
    
    Example:
    POST /evaluate/expression
    {
        "expression": "x + y * z",
        "variables": {"x": 10, "y": 5, "z": 3}
    }
    """
    import uuid
    import time
    
    evaluation_id = str(uuid.uuid4())
    
    try:
        evaluator = SafeExpressionEvaluator(request.variables or {})
        
        start_time = time.time()
        result = evaluator.evaluate(request.expression)
        execution_time_ms = (time.time() - start_time) * 1000
        
        return EvaluationResponse(
            success=True,
            result=result,
            execution_time_ms=execution_time_ms,
            evaluation_id=evaluation_id
        )
        
    except Exception as e:
        return EvaluationResponse(
            success=False,
            error=str(e),
            execution_time_ms=0,
            evaluation_id=evaluation_id
        )

@app.post("/evaluate/code", response_model=EvaluationResponse)
async def evaluate_code(request: EvaluationRequest):
    """
    Execute code in sandboxed environment.
    
    Example:
    POST /evaluate/code
    {
        "code": "def add(a, b): return a + b",
        "test_cases": [
            {"function": "add", "inputs": [2, 3], "expected": 5}
        ]
    }
    """
    import uuid
    
    evaluation_id = str(uuid.uuid4())
    executor = SafeCodeExecutor(timeout_seconds=5)
    
    result = executor.execute(request.code, request.test_cases)
    
    return EvaluationResponse(
        success=result.success,
        output=result.output,
        error=result.error,
        execution_time_ms=result.execution_time_ms,
        evaluation_id=evaluation_id
    )

@app.post("/evaluate/model", status_code=202)
async def evaluate_model(
    request: ModelEvaluationRequest,
    background_tasks: BackgroundTasks
):
    """
    Evaluate AI/ML model (async).
    
    Returns immediately with evaluation ID.
    Results available via polling or webhook.
    """
    import uuid
    
    evaluation_id = str(uuid.uuid4())
    
    # Queue evaluation as background task
    background_tasks.add_task(
        run_model_evaluation,
        evaluation_id=evaluation_id,
        model_id=request.model_id,
        test_dataset=request.test_dataset,
        evaluation_type=request.evaluation_type
    )
    
    return {
        "evaluation_id": evaluation_id,
        "status": "queued",
        "message": "Evaluation started. Check status at /evaluate/status/{evaluation_id}"
    }

@app.get("/evaluate/status/{evaluation_id}")
async def get_evaluation_status(evaluation_id: str):
    """Get status of model evaluation."""
    # Check evaluation status in database/cache
    status = await get_evaluation_status_from_db(evaluation_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Evaluation not found")
    
    return status

async def run_model_evaluation(
    evaluation_id: str,
    model_id: str,
    test_dataset: List[Dict],
    evaluation_type: str
):
    """Background task to run model evaluation."""
    evaluator = ModelEvaluator()
    
    try:
        # Update status
        await update_evaluation_status(evaluation_id, "running")
        
        if evaluation_type == "translation":
            # Extract data
            sources = [item["source"] for item in test_dataset]
            references = [item["reference"] for item in test_dataset]
            
            # Run model inference
            hypotheses = await run_translation_inference(model_id, sources)
            
            # Evaluate
            metrics = evaluator.evaluate_translation(references, hypotheses, sources)
        
        # Save results
        await save_evaluation_results(evaluation_id, metrics)
        await update_evaluation_status(evaluation_id, "completed")
        
    except Exception as e:
        await update_evaluation_status(evaluation_id, "failed", error=str(e))
```

---

## 🎯 Best Practices

### 1. Never Use Raw `eval()` in Production

```python
# ❌ DANGEROUS
user_input = request.json.get("formula")
result = eval(user_input)  # Security vulnerability!

# ✅ SAFE
evaluator = SafeExpressionEvaluator()
result = evaluator.evaluate(user_input)
```

### 2. Always Validate Input

```python
def validate_and_evaluate(expression: str, max_length: int = 1000) -> Any:
    """Validate before evaluation."""
    # Check length
    if len(expression) > max_length:
        raise ValueError("Expression too long")
    
    # Check for dangerous patterns
    dangerous_patterns = [
        '__', 'import', 'exec', 'eval', 'compile',
        'open', 'file', 'input', 'raw_input'
    ]
    
    for pattern in dangerous_patterns:
        if pattern in expression.lower():
            raise ValueError(f"Forbidden pattern: {pattern}")
    
    # Safe evaluation
    return evaluator.evaluate(expression)
```

### 3. Use Timeout and Resource Limits

```python
import signal
from contextlib import contextmanager

@contextmanager
def time_limit(seconds: int):
    """Context manager for timeout."""
    def signal_handler(signum, frame):
        raise TimeoutError("Execution exceeded time limit")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)

# Usage
with time_limit(5):
    result = execute_code(user_code)
```

---

## 📊 Real-World Example: API Quota Evaluation

```python
class QuotaEvaluator:
    """
    Evaluate if API request is within quota limits.
    
    For Bhashini API Marketplace.
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def evaluate_quota(
        self,
        subscription_id: str,
        api_id: str
    ) -> Dict[str, Any]:
        """
        Evaluate if request is within quota.
        
        Returns:
            quota_status with allowed/denied and remaining quota
        """
        # Get subscription limits
        subscription = await self.get_subscription(subscription_id)
        
        if not subscription:
            return {
                "allowed": False,
                "reason": "Invalid subscription",
                "remaining": 0
            }
        
        # Get current usage
        usage_key = f"quota:usage:{subscription_id}:{api_id}"
        current_usage = await self.redis.get(usage_key) or 0
        current_usage = int(current_usage)
        
        # Evaluate
        quota_limit = subscription["quota_limit"]
        remaining = max(0, quota_limit - current_usage)
        
        if current_usage >= quota_limit:
            return {
                "allowed": False,
                "reason": "Quota exceeded",
                "current_usage": current_usage,
                "quota_limit": quota_limit,
                "remaining": 0
            }
        
        return {
            "allowed": True,
            "current_usage": current_usage,
            "quota_limit": quota_limit,
            "remaining": remaining,
            "usage_percentage": (current_usage / quota_limit * 100)
        }
```

---

## 📚 Summary

**Key Takeaways:**

1. **Never use `eval()` with untrusted input** - Always use safe alternatives
2. **AST parsing** - Powerful way to analyze and execute code safely
3. **Rule engines** - Separate business logic from code
4. **Model evaluation** - Comprehensive metrics for AI/ML models
5. **A/B testing** - Statistical rigor for decision making
6. **Sandboxing** - Isolate code execution for security

**For Bhashini Platform:**
- Translation quality evaluation (BLEU, ROUGE)
- Model performance benchmarking
- A/B testing for model versions
- Usage quota evaluation
- Business rules for subscriptions

---

**Complete evaluation systems for production use!** 🎯✨

