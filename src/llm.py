import os
from openai import OpenAI  # 导入OpenAI库用于访问GPT模型
from zhipuai import ZhipuAI # 导入智谱AI库用于访问智谱AI模型
from logger import LOG  # 导入日志模块

class LLM:
    def __init__(self):
        # 创建一个OpenAI客户端实例
        self.openai_client = OpenAI()
        self.zhipu_client = ZhipuAI(api_key=os.getenv('ZHIPU_API_KEY')) 
        # 配置日志文件，当文件大小达到1MB时自动轮转，日志级别为DEBUG
        LOG.add("daily_progress/llm_logs.log", rotation="1 MB", level="DEBUG")

    def generate_daily_report(self, markdown_content, dry_run=False):
        system_prompt = ""
        with open("system_prompt.md", "r", encoding='utf-8') as file:
            system_prompt = file.read()
        
        prompt = markdown_content
        if dry_run:
            # 如果启用了dry_run模式，将不会调用模型，而是将提示信息保存到文件中
            LOG.info("Dry run mode enabled. Saving prompt to file.")
            with open("daily_progress/prompt.txt", "w+") as f:
                f.write(prompt)
            LOG.debug("Prompt saved to daily_progress/prompt.txt")
            return "DRY RUN"

        # 日志记录开始生成报告
        LOG.info("Starting report generation using GPT model.")
        
        try:
            # 调用 OpenAI 模型生成报告
            client = self.openai_client
            # model = "gpt-3.5-turbo"
            model = "gpt-4o"

            # 调用 智谱AI 模型生成报告
            # client = self.zhipu_client
            # model = "glm-4-0520"

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}  # 提交用户角色的消息
                ],
                # temperature=70,
                # max_tokens=4095
            )
            LOG.debug("GPT response: {}", response)
            # 返回模型生成的内容
            return response.choices[0].message.content
        except Exception as e:
            # 如果在请求过程中出现异常，记录错误并抛出
            LOG.error("An error occurred while generating the report: {}", e)
            raise
