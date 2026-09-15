import asyncio
from pathlib import Path

import discord
from discord.bot import Bot
from dotenv import load_dotenv
from ai.intelligence_v3 import Sable

load_dotenv()
bot = Bot()

sable = Sable()

gen_lock = asyncio.Lock()

async def can_reply(message: discord.Message) -> bool:
    # Don't reply to bot messages, including own replies
    if message.author.bot:
        return False

    if isinstance(message.channel, discord.DMChannel):
        return True

    # Don't reply to broadcasts or news-channel messages
    if message.mention_everyone or message.channel.is_news():
        return False

    if message.reference and message.reference.message_id:
        cached_msg = message.reference.cached_message

        if cached_msg:
            original_msg = cached_msg
        else:
            try:
                channel = bot.get_channel(message.reference.channel_id)
                original_msg = await channel.fetch_message(
                    message.reference.message_id
                )
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                original_msg = None

        if original_msg and original_msg.author.id == bot.user.id:
            return True

    # Reply only when SABLE or one of her roles is mentioned
    bot_mentioned = message.channel.me in message.mentions
    role_mentioned = any(
        role in message.role_mentions
        for role in message.guild.me.roles
    )

    return bot_mentioned or role_mentioned

@bot.event
async def on_message(message: discord.Message):
    if not await can_reply(message):
        return
    
    workspace = Path(__file__).parents[1] / 'file-workspace' / 'discord-attachments'
    workspace.mkdir(parents=True, exist_ok=True)
    
    attachment_paths = []
    
    for attach in message.attachments:
        if attach.content_type not in {'text/plain', 'text/markdown', 'text/csv', 'text/xml', 'application/json'}:
            continue
        
        path = workspace / f'{message.author.name}-{attach.filename}'
        
        await attach.save(path, use_cached=True)
        attachment_paths.append(path)
    
    if not sable.submit(message.clean_content, attachment_paths):
        return
    
    async with gen_lock, message.channel.typing():
        rsp = sable.generate()
    
    await message.reply(rsp)

@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")
    
@bot.slash_command(name="helloworld", description="response test")
async def hello(ctx: discord.ApplicationContext):
    await ctx.respond("Hey!")