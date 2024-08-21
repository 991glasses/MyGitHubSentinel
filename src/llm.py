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
        # 构建一个用于生成报告的提示文本，要求生成的报告包含新增功能、主要改进和问题修复
        system_prompt = "以下是一个 git 项目的最新进展，请根据功能合并同类项，形成一份简报，至少包含：1）新增功能；2）主要改进；3）修复问题。\n你可以根据开头的标题判断出该项目是否是一个知名项目，如果是，请根据你了解到的知识判断哪些改动项是重要的，将其高亮标注出来。\n简报使用中文撰写。"
        
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
            # client = self.openai_client
            # model = "gpt-3.5-turbo"

            # 调用 智谱AI 模型生成报告
            client = self.zhipu_client
            model = "glm-4-0520"

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}  # 提交用户角色的消息
                ]
            )
            LOG.debug("GPT response: {}", response)
            # 返回模型生成的内容
            return response.choices[0].message.content
        except Exception as e:
            # 如果在请求过程中出现异常，记录错误并抛出
            LOG.error("An error occurred while generating the report: {}", e)
            raise
