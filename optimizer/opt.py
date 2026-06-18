import os
import json
import re
from openai import OpenAI
from PIL import Image
import base64
from io import BytesIO

from dotenv import load_dotenv

load_dotenv()


class VisualPromptOptimizer:
    """
    视觉提示优化器 - 基于训练集和验证集自动优化视觉分类prompt
    参考 optimizer.md 描述的迭代优化流程
    """
    
    def __init__(self, base_url, api_key, base_model,
                 judge_base_url=None, judge_api_key=None, judge_model=None):
        self.base_client = OpenAI(base_url=base_url, api_key=api_key)
        self.base_model = base_model
        
        # Judge model 默认与 base model 相同
        judge_base_url = judge_base_url or base_url
        judge_api_key = judge_api_key or api_key
        self.judge_client = OpenAI(base_url=judge_base_url, api_key=judge_api_key)
        self.judge_model = judge_model or base_model
        
        self.temperature = 0.2
        self.result_pattern = r'<result>(.*?)</result>'
        self.threshold = 0.95
        self.negative_feedback_list = []
        
    def _image_to_base64(self, image_path):
        """将图像转为 base64"""
        image = Image.open(image_path).convert('RGB')
        buffer = BytesIO()
        image.save(buffer, format='JPEG')
        b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f'data:image/jpeg;base64,{b64}'
    
    def _infer_chat_completion(self, system_prompt, image_path):
        """对单张图像进行推理"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": self._image_to_base64(image_path)}},
                {"type": "text", "text": "请按照prompt要求分析该图像"}
            ]}
        ]
        
        response = self.base_client.chat.completions.create(
            model=self.base_model,
            messages=messages,
            temperature=self.temperature,
            extra_body={"chat_template_kwargs": {"enable_thinking": True}}
        )
        return response.choices[0].message.content
    
    def _infer_response(self, system_prompt, image_path):
        messages = []
        image_b64 = self._image_to_base64(image_path)
        image_input = {
            "type": "input_image",
            "image_url": image_b64
        }
        text_input = {"type": "input_text", "text": "请按照prompt要求分析该图像"}
        messages.append(
            {"role": "user", "content": [text_input, image_input]}
        )
        response = self.base_client.responses.create(
            model=self.base_model,
            instructions=system_prompt,
            input=messages,
            temperature=self.temperature,
            reasoning={"effort": "medium"},
            extra_body={"thinking": {"type": "enabled"}},
        )
        return f"{response.output[0].summary[0].text}, {response.output_text}"

    
    def _evaluate_dataset(self, system_prompt, data_dict):
        """在数据集上运行推理，返回结果列表"""
        results = []
        for idx, (img_path, label) in enumerate(data_dict.items()):
            output = self._infer_response(system_prompt, img_path)
            match = re.search(self.result_pattern, output, re.DOTALL)
            pred = match.group(1).strip() if match else output.strip()
            results.append({
                'img_path': img_path,
                'label': label,
                'pred': pred,
                'full_output': output
            })
            print(f'finish processing idx {idx}')
            if idx > 30:    # removal
                break
        return results
    
    def _calc_accuracy(self, results):
        """计算准确率"""
        correct = sum(1 for r in results if r['label'] == r['pred'])
        return correct / len(results) if results else 0.0
    
    def _get_errors(self, results):
        """获取错误case"""
        return [r for r in results if r['label'] != r['pred']]
    
    def _analyze_errors(self, errors, current_prompt, negative_feedback):
        """
        使用Judge模型分析错误原因，结合负反馈列表给出优化建议
        """
        if not errors:
            return []
        
        # 构建错误案例描述（限制数量避免prompt过长）
        error_text = "\n".join([
            f"- 图片: {e['img_path']}\n  真实标签: {e['label']}\n  预测: {e['pred']}\n  输出: {e['full_output'][:500]}"
            for e in errors[:10]
        ])
        
        fb_text = "\n".join([f"- {fb}" for fb in negative_feedback]) if negative_feedback else "暂无"
        
        judge_prompt = f"""你是视觉提示优化专家。请分析以下错误案例，结合当前prompt和负反馈列表，给出优化建议。

## 当前Prompt
{current_prompt[:4000]}

## 负反馈列表（已验证会导致性能下降的方向，请避免）
{fb_text}

## 错误案例（共{len(errors)}个，展示前10个）
{error_text}

请按以下格式输出每条优化建议：
<suggestion>
标题: xxx
说明: xxx
</suggestion>

给出3-5条具体、可操作的优化建议。
"""
        response = self.judge_client.chat.completions.create(
            model=self.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=self.temperature,
        )
        output = response.choices[0].message.content
        suggestions = re.findall(r'<suggestion>(.*?)</suggestion>', output, re.DOTALL)
        return [s.strip() for s in suggestions]
    
    def _apply_suggestion(self, current_prompt, suggestion):
        """
        根据单条优化建议生成修改后的prompt
        """
        apply_prompt = f"""你是视觉提示优化专家。请根据以下优化建议修改prompt。

## 原始Prompt
{current_prompt}

## 优化建议
{suggestion}

请输出修改后的完整prompt。要求：
1. 保持原有结构和格式
2. 只针对优化建议进行修改
3. 确保修改逻辑严谨，不引入新矛盾
"""
        response = self.judge_client.responses.create(
            model=self.judge_model,
            input=apply_prompt,
            temperature=self.temperature,
        )
        return response.output_text
    
    def _merge_suggestions(self, current_prompt, accepted_prompts):
        """
        合并多条采纳的优化建议，生成新的prompt
        """
        if len(accepted_prompts) == 1:
            return accepted_prompts[0]
        
        versions_text = "\n\n".join([
            f"优化版本{i+1}:\n{p[:2000]}..."
            for i, p in enumerate(accepted_prompts)
        ])
        
        merge_prompt = f"""你是视觉提示优化专家。请将以下多个优化版本整合为一个prompt。

## 原始Prompt
{current_prompt[:2000]}...

## 待整合的优化版本
{versions_text}

请输出整合后的完整prompt。要求：
1. 保持原有结构和格式
2. 合理整合所有优化点，避免冲突
3. 确保整合后的prompt逻辑一致
"""
        response = self.judge_client.responses.create(
            model=self.judge_model,
            input=merge_prompt,
            temperature=self.temperature,
        )
        return response.output_text
    
    def optimize(self, train_file, validate_file, initial_prompt_file,
                 max_iterations=10, output_dir=None):
        """
        主优化循环
        
        流程:
        1. 使用base模型和当前prompt，对训练集和验证集全量数据推理，统计精度
        2. 收集训练集错误case，使用Judge模型分析原因，结合负反馈列表生成优化建议
        3. 对每条建议，在验证集上验证精度提升情况
           - 有提升 -> 标记为待采纳
           - 劣化 -> 加入负反馈列表
        4. 合并所有待采纳建议，生成新prompt
        5. 重复直到验证集精度不再提升或达到最大迭代次数
        """
        # 加载数据
        with open(train_file, 'r') as f:
            train_data = json.load(f)
        with open(validate_file, 'r') as f:
            validate_data = json.load(f)
        with open(initial_prompt_file, 'r') as f:
            current_prompt = f.read()
        
        best_prompt = current_prompt
        best_val_acc = 0.0
        best_train_acc = 0.0
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for iteration in range(max_iterations):
            print(f"\n{'='*50}")
            print(f"迭代 {iteration+1}/{max_iterations}")
            print(f"{'='*50}")
            
            # Step 1: 全量推理，统计训练集和验证集精度
            print("评估训练集...")
            train_results = self._evaluate_dataset(current_prompt, train_data)
            train_acc = self._calc_accuracy(train_results)
            print(f"训练集精度: {train_acc:.4f}")
            
            print("评估验证集...")
            val_results = self._evaluate_dataset(current_prompt, validate_data)
            val_acc = self._calc_accuracy(val_results)
            print(f"验证集精度: {val_acc:.4f}")
            
            # 检查是否达到阈值
            if val_acc >= self.threshold:
                print(f"达到阈值 {self.threshold}, 停止优化")
                break
            
            # 更新最优
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_train_acc = train_acc
                best_prompt = current_prompt
                print(f"更新最优模型: 验证集={best_val_acc:.4f}, 训练集={best_train_acc:.4f}")
            
            # Step 2: 收集错误case，使用Judge模型分析并生成优化建议
            errors = self._get_errors(train_results)
            print(f"训练集错误: {len(errors)}/{len(train_results)}")
            
            if not errors:
                print("训练集无错误，停止优化")
                break
            
            print("分析错误并生成优化建议...")
            suggestions = self._analyze_errors(errors, current_prompt, 
                                              self.negative_feedback_list)
            print(f"生成 {len(suggestions)} 条建议")
            
            if not suggestions:
                print("无优化建议，停止优化")
                break
            
            # Step 3: 对每条建议进行验证
            accepted = []
            for i, suggestion in enumerate(suggestions):
                print(f"验证建议 {i+1}/{len(suggestions)}...")
                new_prompt = self._apply_suggestion(current_prompt, suggestion)
                new_val_results = self._evaluate_dataset(new_prompt, validate_data)
                new_val_acc = self._calc_accuracy(new_val_results)
                print(f"  验证集精度: {new_val_acc:.4f} (当前: {val_acc:.4f})")
                
                if new_val_acc > val_acc:
                    accepted.append(new_prompt)
                    print(f"  -> 采纳")
                else:
                    fb = f"建议{i+1} ({suggestion[:100]}...) 精度从{val_acc:.4f}降至{new_val_acc:.4f}"
                    self.negative_feedback_list.append(fb)
                    print(f"  -> 拒绝 (加入负反馈列表)")
            
            # Step 4: 合并所有待采纳建议，生成新prompt
            if accepted:
                print(f"合并 {len(accepted)} 条采纳的建议...")
                current_prompt = self._merge_suggestions(current_prompt, accepted)
                
                if output_dir:
                    save_path = os.path.join(output_dir, f"prompt_iter_{iteration+1}.md")
                    with open(save_path, 'w', encoding='utf-8') as f:
                        f.write(current_prompt)
                    print(f"保存prompt到: {save_path}")
            else:
                print("无建议被采纳，停止优化")
                break
        
        # 输出最终结果
        print(f"\n{'='*50}")
        print(f"优化完成")
        print(f"{'='*50}")
        print(f"最优验证集精度: {best_val_acc:.4f}")
        print(f"对应训练集精度: {best_train_acc:.4f}")
        
        if output_dir:
            final_path = os.path.join(output_dir, "final_prompt.md")
            with open(final_path, 'w', encoding='utf-8') as f:
                f.write(best_prompt)
            print(f"最终prompt保存到: {final_path}")
            
            summary = {
                'best_val_accuracy': best_val_acc,
                'best_train_accuracy': best_train_acc,
                'negative_feedback_list': self.negative_feedback_list
            }
            summary_path = os.path.join(output_dir, "optimization_summary.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"优化摘要保存到: {summary_path}")
        
        return best_prompt, best_val_acc


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual Prompt Optimizer")
    parser.add_argument("--base_url", type=str, default=os.getenv("SPEECH_BASE_URL"), help="Base模型 API URL")
    parser.add_argument("--api_key", type=str, default=os.getenv("SPEECH_API_KEY"), help="API key")
    parser.add_argument("--base_model", type=str, default=os.getenv("SPEECH_MODEL"), help="Base模型名称")
    parser.add_argument("--judge_url", type=str, default=os.getenv("API_URL"), help="Judge模型 API URL (默认同base)")
    parser.add_argument("--judge_key", type=str, default=os.getenv("API_KEY"), help="Judge API key (默认同base)")
    parser.add_argument("--judge_model", type=str, default=os.getenv("MODEL_NAME"), help="Judge模型名称 (默认同base)")
    parser.add_argument("--train_file", type=str, required=True, help="训练集JSON文件路径")
    parser.add_argument("--val_file", type=str, required=True, help="验证集JSON文件路径")
    parser.add_argument("--prompt_file", type=str, required=True, help="初始prompt文件路径")
    parser.add_argument("--max_iterations", type=int, default=10, help="最大迭代次数")
    parser.add_argument("--output_dir", type=str, default='exps/outputs', help="输出目录")
    parser.add_argument("--threshold", type=float, default=0.95, help="目标精度阈值")
    
    args = parser.parse_args()
    
    optimizer = VisualPromptOptimizer(
        base_url=args.base_url,
        api_key=args.api_key,
        base_model=args.base_model,
        judge_base_url=args.judge_url,
        judge_api_key=args.judge_key,
        judge_model=args.judge_model,
    )
    optimizer.threshold = args.threshold
    
    optimizer.optimize(
        train_file=args.train_file,
        validate_file=args.val_file,
        initial_prompt_file=args.prompt_file,
        max_iterations=args.max_iterations,
        output_dir=args.output_dir,
    )
