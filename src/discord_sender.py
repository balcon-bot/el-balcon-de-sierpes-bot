import discord
import os
from dotenv import load_dotenv

load_dotenv()

intents = discord.Intents.default()
client = discord.Client(intents=intents)

@client.event
async def on_ready():
    channel_id = int(os.getenv('DISCORD_CHANNEL_ID'))
    channel = client.get_channel(channel_id)

    if not os.path.exists("last_post_data.txt"):
        await client.close()
        return

    try:
        with open("last_post_data.txt", "r", encoding="utf-8") as f:
            lines = f.readlines()
            text = lines[0].strip()
            img_path = lines[1].strip()

        if os.path.exists(img_path):
            file = discord.File(img_path, filename="cartel.png")

            embed = discord.Embed(
                description=text,
                color=0x800080
            )

            embed.set_image(url="attachment://cartel.png")
            embed.set_footer(text="El Balcón de Sierpes | Bot Cofrade")

            msg = await channel.send(file=file, embed=embed)

            with open("last_msg_id.txt", "w") as out:
                out.write(str(msg.id))

            print("✅ Publicado en Discord.")

        else:
            print("❌ Imagen no encontrada.")

    except Exception as e:
        print(f"❌ Error en Discord: {e}")

    await client.close()

client.run(os.getenv('DISCORD_TOKEN'))
