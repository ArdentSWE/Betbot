import os
import asyncio
import httpx
import discord
from discord.ext import commands, tasks
from datetime import datetime
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_ai import Agent

# Load environment variables (.env file for local, Railway Variables for cloud)
load_dotenv()

# ==========================================
# BOT CONFIGURATION & ACE BRANDING
# ==========================================
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

ACE_COLOR = 0xD4AF37 # Premium Gold hex code
ACE_FOOTER = "ACE | Omni-Factor Quantitative Terminal"

# Initialize Discord Bot with the command prefix '!ace '
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!ace ", intents=intents)

# Pydantic-AI v2.0+ architecture. 
# Explicitly routes to Chat Completions and auto-detects OPENAI_API_KEY from environment variables.
MODEL = 'openai-chat:gpt-5.5'


# ==========================================
# LAYER 4: STRICT OUTPUT SERIALIZATION
# ==========================================
class AlphaPlay(BaseModel):
    game_identifier: str = Field(description="The matchup name, e.g., 'LAD @ PIT'")
    sport_league: str = Field(description="The league name, e.g., 'MLB', 'NHL', 'NBA', 'UFC'")
    market_position: str = Field(description="The precise betting market, e.g., 'LAD F5 ML' or 'Player Prop'")
    current_odds: str = Field(description="Consensus odds")
    confidence_rating: float = Field(description="Strict scale from 1.0 to 10.0. Only 8.5+ allowed.")
    data_grounded_why: str = Field(description="Concise, bulleted justification.")
    devils_advocate_refutation: str = Field(description="The explicit breakdown of how the play survives the reverse-test.")   

    @field_validator('confidence_rating')    
    @classmethod
    def enforce_alpha_threshold(cls, v: float) -> float:
        if v < 8.5:
            raise ValueError("Play does not clear the strict 8.5+ ACE threshold.")
        return v


# ==========================================
# LAYER 1: WEB INGESTION ENGINE
# ==========================================
class WebScrapingIngestionEngine:
    def __init__(self):
        self.client = httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"})

    async def fetch_live_slate(self, sport: str, league: str, target_date: str = None) -> list:
        """
        Scrapes ESPN public endpoints.
        Accepts an optional target_date (e.g., '2026-06-15') for forward/backward scanning.
        """
        base_url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
        
        if target_date:
            clean_date = target_date.replace("-", "")
            url = f"{base_url}?dates={clean_date}"
        else:
            url = base_url

        try:
            response = await self.client.get(url, timeout=15.0)
            if response.status_code != 200: return []
            
            data = response.json()
            extracted_games = []

            for event in data.get("events", []):
                comp = event.get("competitions", [{}])[0]
                teams = comp.get("competitors", [])
                away = next((t.get("team", {}).get("abbreviation") for t in teams if t.get("homeAway") == "away"), "AWAY")
                home = next((t.get("team", {}).get("abbreviation") for t in teams if t.get("homeAway") == "home"), "HOME")
                
                odds_record = comp.get("odds", [{}])[0] if comp.get("odds") else {}
                
                extracted_games.append({
                    "game_name": f"{away} @ {home}",
                    "league": league.upper(),
                    "venue": comp.get("venue", {}).get("fullName", "Unknown"),
                    "market_line": odds_record.get("details", "N/A"),
                    "total": odds_record.get("overUnder", 0.0),
                    "raw_meta": str(event)[:1500] 
                })
            return extracted_games
        except Exception:
            return []


# ==========================================
# LAYERS 2 & 3: MULTI-AGENT PIPELINE (V2.0)
# ==========================================

structurer_agent = Agent(
    model=MODEL,
    system_prompt="Normalize the raw scraped sports JSON into a unified, crisp Markdown block mapping rosters, injuries, and stadium conditions.",
    model_settings={'temperature': 0.1}
)

analyst_agent = Agent(
    model=MODEL,
    system_prompt=(
        "THE UNIVERSAL OMNI-FACTOR MASTER PROMPT (V2.0)\n\n"
        "STAGE 1: MARKET SCRAPE & SITUATIONAL PHYSICS\n"
        "Context: Identify the exact date, location, real-time odds, implied probabilities, and public betting distribution.\n"
        "Environmental Physics: Quantify the stadium dimensions, weather, altitude, and travel fatigue.\n\n"
        "STAGE 2: ROSTER GEOMETRY, DEPTH & COHERENCE\n"
        "Pitching/Asset Disparity: Isolate the exact ERA, WHIP, K/9, and recent performance metrics.\n"
        "Injury & Roster Disruption: Calculate the specific offensive and defensive vacuums created by missing personnel and how they alter structural geometry.\n\n"
        "STAGE 3: MICRO-PROP INDUCTION\n"
        "Volatility & Clustering: Build the game from the bottom up. Identify which player props possess the highest mathematical floors based on the starting matchup.\n\n"
        "STAGE 4: MACRO-MARKET DEDUCTION\n"
        "Market Translation: Map micro-prop data onto macro markets. Identify where bookmakers are over-leveraging team records while ignoring underlying metrics.\n\n"
        "STAGE 5: THE DEVIL'S ADVOCATE PROTOCOL\n"
        "Stress-Test the Play: Actively try to break your own logic. Assume the worst-case scenario.\n"
        "Potential Issues: Identify historical vulnerabilities, bullpen collapse probabilities, or negative momentum.\n"
        "The Refutation: If the mathematical edge does not completely absorb and neutralize these risks, discard the play.\n\n"
        "STAGE 6: THE ANTI-HALLUCINATION PROTOCOL\n"
        "Strict Data Grounding: Mandate all data is pulled from verified databases.\n"
        "Narrative Scrubbing: Completely eliminate gut feelings and unverified momentum narratives.\n\n"
        "STAGE 7: FINAL VERDICT & CONSERVATIVE EXECUTION\n"
        "Assign a conservative confidence rating on a scale of 1.0 to 10.0. Only plays hitting 8.5+ pass."
    ),
    model_settings={'temperature': 0.1}
)

validator_agent = Agent(
    model=MODEL,
    output_type=AlphaPlay, # Fixed Pydantic-AI v2.0 parameter
    system_prompt="Cast the analytical breakdown into the JSON schema. Verify the confidence rating is strictly 8.5+. Strip narrative fluff.",
    model_settings={'temperature': 0.1}
)


# ==========================================
# DISCORD COMMAND & CONTINUOUS LOOP
# ==========================================
async def scan_and_process_slate(ctx=None, channel=None, target_date: str = None):
    """The master continuous loop that processes games one by one."""
    out_channel = ctx.channel if ctx else channel
    
    display_date = target_date if target_date else datetime.now().strftime('%Y-%m-%d')
    
    embed = discord.Embed(
        title="⚡ ACE TERMINAL PROTOCOL INITIATED",
        description=f"Scraping datasets for **{display_date}**. Locking sequence engaged...",
        color=ACE_COLOR
    )
    embed.set_footer(text=ACE_FOOTER)
    status_msg = await out_channel.send(embed=embed)

    scraper = WebScrapingIngestionEngine()
    slates = [
        ("baseball", "mlb"), ("basketball", "nba"), 
        ("hockey", "nhl"), ("mma", "ufc")
    ]
    
    all_games = []
    for sport, league in slates:
        all_games.extend(await scraper.fetch_live_slate(sport, league, target_date))

    if not all_games:
        await out_channel.send(f"No active markets found for {display_date}.")
        return

    found_plays = 0
    
    for game in all_games:
        await status_msg.edit(embed=discord.Embed(
            title="🔄 ACE TERMINAL SCANNING",
            description=f"**Processing:** {game['game_name']} ({game['league']})\nRunning Omni-Factor V2.0...",
            color=ACE_COLOR
        ).set_footer(text=ACE_FOOTER))
        
        try:
            struct_output = await structurer_agent.run(str(game))
            analysis_output = await analyst_agent.run(struct_output.data)
            validated_payload = await validator_agent.run(analysis_output.data)
            
            play = validated_payload.data
            found_plays += 1
            
            # Format high-quality play with ACE branding
            play_embed = discord.Embed(
                title=f"🎯 ACE EXCLUSIVE | {play.sport_league} ALPHA",
                color=ACE_COLOR
            )
            play_embed.add_field(name="Matchup", value=f"**{play.game_identifier}**", inline=False)
            play_embed.add_field(name="Market Position", value=f"`{play.market_position}`", inline=True)
            play_embed.add_field(name="Odds", value=play.current_odds, inline=True)
            play_embed.add_field(name="Confidence", value=f"**{play.confidence_rating}/10.0**", inline=True)
            play_embed.add_field(name="The Why", value=play.data_grounded_why, inline=False)
            play_embed.add_field(name="Devil's Advocate Protocol", value=f"*{play.devils_advocate_refutation}*", inline=False)
            play_embed.set_footer(text=ACE_FOOTER)
            
            await out_channel.send(embed=play_embed)
            
        except Exception:
            # Game discarded (Failed 8.5 threshold or logic was broken in validation layer)
            pass

    # Final wrap-up
    summary = discord.Embed(
        title="🏁 ACE TERMINAL SEQUENCE COMPLETE",
        description=f"Scan finished. Isolated **{found_plays}** alpha plays clearing the 8.5 threshold for {display_date}.",
        color=ACE_COLOR
    )
    summary.set_footer(text=ACE_FOOTER)
    await status_msg.edit(embed=summary)


@bot.command(name="scan")
async def manual_scan(ctx, target_date: str = None):
    """
    Triggers the terminal interactively.
    Usage 1: !ace scan (Scrapes today's slate)
    Usage 2: !ace scan 2026-06-15 (Scrapes a specific future or past date)
    """
    await scan_and_process_slate(ctx=ctx, target_date=target_date)


@tasks.loop(hours=24)
async def automated_daily_scan():
    """Background loop that runs once every 24 hours."""
    channel_id_str = os.getenv("DISCORD_CHANNEL_ID")
    if not channel_id_str: 
        return
    
    try:
        channel_id = int(channel_id_str)
        channel = bot.get_channel(channel_id)
        if channel:
            await scan_and_process_slate(channel=channel)
    except ValueError:
        print("Error: DISCORD_CHANNEL_ID must be a valid integer.")


@bot.event
async def on_ready():
    print(f"==========================================")
    print(f" ACE TERMINAL LOGGED IN AS: {bot.user.name}")
    print(f"==========================================")
    if not automated_daily_scan.is_running():
        automated_daily_scan.start()

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("CRITICAL: DISCORD_BOT_TOKEN missing.")
    else:
        bot.run(DISCORD_TOKEN)