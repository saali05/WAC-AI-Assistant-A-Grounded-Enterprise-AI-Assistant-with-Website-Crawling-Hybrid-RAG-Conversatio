# WAC AI Assistant --- Analytics and Usage

## Purpose

The analytics subsystem records AI usage so the application can measure
model consumption, latency, estimated cost, and voice usage.

## Usage Information

`AIUsage` can represent:

-   Provider
-   Model
-   Request type
-   Input tokens
-   Output tokens
-   Total tokens
-   Cached tokens
-   Thinking tokens
-   Estimated cost
-   Currency
-   Latency
-   Tokens per second
-   Context limit
-   Remaining context
-   Audio input duration
-   Audio output duration
-   Live session ID
-   Usage source
-   Quota scope

## Text Usage Flow

``` text
User Request
     |
AI Provider
     |
Provider Response
     |
Usage Metadata
     |
AIUsage
     |
UsageService
     |
MongoDB
     |
Analytics Dashboard
```

The Gemini provider can extract provider usage metadata including
prompt, output, total, cached, and thinking token counts when available.

## Latency

The provider measures the elapsed time for an AI request.

``` text
Start
  |
Gemini API
  |
Response
  |
End
```

Latency is stored in milliseconds. When output token information is
available, tokens per second can also be calculated.

## Cost Estimation

The project contains model pricing configuration and calculates an
estimated cost.

``` text
Model + Input Usage + Output Usage
             |
             v
       calculate_cost()
             |
             v
       Estimated USD Cost
```

For voice usage, audio input and output usage can also be included.

The estimate is application-side pricing logic and should not be treated
as the provider's authoritative billing statement.

## Context Usage

When a context limit is known:

``` text
Context Limit - Input Tokens = Context Remaining
```

This is useful for monitoring how much model context remains.

## Voice Analytics

Voice usage can include:

-   Input audio seconds
-   Output audio seconds
-   Input tokens
-   Output tokens
-   Latency
-   Live session ID
-   Estimated cost

The voice API passes these values to `UsageService`.

## Provider Tracking

Usage records include provider and model, allowing the application to
distinguish Gemini, Groq, and Gemini Live usage.

## Why Analytics Is Important

Analytics helps measure:

-   Request volume
-   Token consumption
-   Estimated cost
-   Latency
-   Provider/model usage
-   Voice consumption
-   Session resource usage

## Quota vs Application Analytics

These are different:

``` text
Application analytics
    |
    v
Local usage/cost records

Provider
    |
    v
Actual API quota and rate limits
```

A provider can return a 429 quota error even if local analytics does not
show the same value as provider billing.

## Rate Limits

Gemini quota failures such as `429 RESOURCE_EXHAUSTED` are provider-side
rate/quota conditions and should be handled separately from successful
usage records.
