package org.llm4s.samples.basic

import org.llm4s.llmconnect._
import org.llm4s.llmconnect.model._
import org.llm4s.config.Llm4sConfig
import org.slf4j.LoggerFactory

/**
 * Provider fallback example demonstrating multi-provider support in LLM4S.
 *
 * This example shows:
 * - Configuring multiple LLM providers in code
 * - Attempting a request across providers using fallback logic
 * - Running the same prompt across providers without changing application code
 * - Using the first provider that successfully generates a response
 *
 * == Quick Start ==
 *
 * 1. This example demonstrates provider fallback using providers
 *    configured directly in `providerConfigs`. Providers are tried
 *    in the order listed there (not via `LLM_MODEL`).
 *
 * 2. (Optional) Set API keys for cloud providers:
 *    {{{
 *    export OPENAI_API_KEY=sk-...
 *    export ANTHROPIC_API_KEY=sk-ant-...
 *    }}}
 *
 *    If API keys are not provided, LLM4S will automatically
 *    fall back to the next available provider (e.g. Ollama).
 *    Ensure that ollama is running locally if you intend to use it.
 *
 * 3. Run the example:
 *    {{{
 *    sbt "samples/runMain org.llm4s.samples.basic.ProviderFallbackExample"
 *    }}}
 *
 * == Expected Output ==
 * The example prints the model/provider that successfully handled the request,
 * followed by the generated response. If earlier providers fail due to missing
 * configuration or network errors, fallback occurs transparently.
 *
 * == Supported Providers ==
 * - '''OpenAI''': `LLM_MODEL=openai/<model>`, requires `OPENAI_API_KEY`
 * - '''Anthropic''': `LLM_MODEL=anthropic/<model>`, requires `ANTHROPIC_API_KEY`
 * - '''Ollama''': `LLM_MODEL=ollama/<model>`, no API key required (local)
 */

object ProviderFallbackExample {

  private val logger = LoggerFactory.getLogger(getClass)

  def main(args: Array[String]): Unit = {

    val result = for {
      providerCfg <- Llm4sConfig.provider()
      client      <- LLMConnect.getClient(providerCfg)

      completion <- client.complete(
        Conversation(
          Seq(UserMessage("Hello, world! Which provider am I talking to?"))
        )
      )

      _ = logger.info(s"[SUCCESS] ${completion.message.content}")
    } yield ()

    result.fold(
      err => logger.error("{}", err.formatted),
      identity
    )
  }
}
