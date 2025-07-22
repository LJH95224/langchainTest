import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent

def get_weather(city: str) -> dict:
    """获取指定城市的当前天气报告。

    Args:
        city (str): 需要获取天气报告的城市名称。

    Returns:
        dict: 包含状态和结果或错误信息的字典。
    """
    print(f"调用了get_weather函数，城市名称: '{city}'")  # 添加调试信息
    
    # 城市天气数据字典 - 在实际应用中应该连接到真实的天气API
    city_weather = {
        "new york": {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."
        },
        "newyork": {
            "status": "success",
            "report": "The weather in New York is sunny with a temperature of 25 degrees Celsius (77 degrees Fahrenheit)."
        },
        "纽约": {
            "status": "success",
            "report": "纽约天气晴朗，气温25摄氏度（77华氏度）。"
        },
        "shanghai": {
            "status": "success",
            "report": "The weather in Shanghai is cloudy with a temperature of 22 degrees Celsius (72 degrees Fahrenheit)."
        },
        "shang hai": {
            "status": "success",
            "report": "The weather in Shanghai is cloudy with a temperature of 22 degrees Celsius (72 degrees Fahrenheit)."
        },
        "上海": {
            "status": "success",
            "report": "上海今天多云，气温22摄氏度（72华氏度）。"
        },
        "beijing": {
            "status": "success",
            "report": "The weather in Beijing is sunny with a temperature of 24 degrees Celsius (75 degrees Fahrenheit)."
        },
        "北京": {
            "status": "success",
            "report": "北京今天晴朗，气温24摄氏度（75华氏度）。"
        },
        "london": {
            "status": "success",
            "report": "The weather in London is rainy with a temperature of 18 degrees Celsius (64 degrees Fahrenheit)."
        },
        "伦敦": {
            "status": "success",
            "report": "伦敦今天下雨，气温18摄氏度（64华氏度）。"
        },
        "tokyo": {
            "status": "success",
            "report": "The weather in Tokyo is partly cloudy with a temperature of 26 degrees Celsius (79 degrees Fahrenheit)."
        },
        "东京": {
            "status": "success",
            "report": "东京今天多云间晴，气温26摄氏度（79华氏度）。"
        }
    }
    
    city_lower = city.lower()
    print(f"转换为小写后的城市名称: '{city_lower}'")  # 添加调试信息
    print(f"城市字典中的键: {list(city_weather.keys())}")  # 添加调试信息
    
    if city_lower in city_weather:
        print(f"找到城市'{city_lower}'的天气信息")  # 添加调试信息
        return city_weather[city_lower]
    else:
        print(f"未找到城市'{city_lower}'的天气信息")  # 添加调试信息
        return {
            "status": "error",
            "error_message": f"Weather information for '{city}' is not available."
        }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city (str): The name of the city for which to retrieve the current time.

    Returns:
        dict: status and result or error msg.
    """
    # 城市到时区的映射字典
    city_to_timezone = {
        "new york": "America/New_York",
        "newyork": "America/New_York",
        "纽约": "America/New_York",
        "shanghai": "Asia/Shanghai",
        "shang hai": "Asia/Shanghai",
        "上海": "Asia/Shanghai",
        "beijing": "Asia/Shanghai", 
        "北京": "Asia/Shanghai",
        "london": "Europe/London",
        "伦敦": "Europe/London",
        "tokyo": "Asia/Tokyo",
        "东京": "Asia/Tokyo"
    }
    
    city_lower = city.lower()
    if city_lower in city_to_timezone:
        tz_identifier = city_to_timezone[city_lower]
    else:
        return {
            "status": "error",
            "error_message": (
                f"Sorry, I don't have timezone information for {city}."
            ),
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    report = (
        f'The current time in {city} is {now.strftime("%Y-%m-%d %H:%M:%S %Z%z")}'
    )
    return {"status": "success", "report": report}


root_agent = Agent(
    name="weather_time_agent",
    model="gemini-2.0-flash",
    description=(
        "Agent to answer questions about the time and weather in a city."
    ),
    instruction=(
        "You are a helpful agent who can answer user questions about the time and weather in a city. "
        "When users ask about weather, extract the city name and call the get_weather function. "
        "For Shanghai/上海, Beijing/北京, Tokyo/东京, London/伦敦, and New York/纽约, you should have data. "
        "When users ask about time, extract the city name and call the get_current_time function. "
        "Always present the information from these tools in a friendly manner."
    ),
    tools=[get_weather, get_current_time],
)