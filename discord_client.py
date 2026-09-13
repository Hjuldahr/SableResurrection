import discord
from dotenv import load_dotenv

load_dotenv()
bot = discord.Bot()

@bot.event
async def on_ready():
    print(f"{bot.user} is ready and online!")
    
@bot.slash_command(name="helloworld", description="response test")
async def hello(ctx: discord.ApplicationContext):
    await ctx.respond("Hey!")