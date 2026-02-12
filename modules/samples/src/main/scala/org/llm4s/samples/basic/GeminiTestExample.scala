package org.llm4s.samples.basic

import org.llm4s.llmconnect.LLMConnect
import org.llm4s.llmconnect.model._
import org.llm4s.toolapi._
import org.llm4s.toolapi.tools.WeatherTool

/**
 * Test example for Google Gemini provider.
 * Tests simple completion, streaming, and tool calling.
 *
 * Run with:
 * {{{
 * sbt "samples/runMain org.llm4s.samples.basic.GeminiTestExample"
 * }}}
 */

object GeminiTestExample {

  def main(args: Array[String]): Unit = {

    val result = for {
      providerCfg <- org.llm4s.config.Llm4sConfig.provider()
      client      <- LLMConnect.getClient(providerCfg)
      _ = {
        println("=" * 60)
        println("Gemini Provider Test Suite")
        println("=" * 60)

        testSimpleCompletion(client)
        testStreaming(client)
        testToolCalling(client)

        println("=" * 60)
        println("All tests complete!")
        println("=" * 60)
      }
    } yield ()

    result.fold(
      err => println(s"Configuration error: ${err.formatted}"),
      identity
    )
  }

  private def testSimpleCompletion(client: org.llm4s.llmconnect.LLMClient): Unit = {

    val conversation = Conversation(
      Seq(
        SystemMessage("You are a helpful assistant. Be concise."),
        UserMessage("What is 2 + 2? Answer with just the number.")
      )
    )

    client.complete(conversation) match {
      case Right(completion) =>
        println(" Simple completion SUCCESS")
        println(s"   Model: ${completion.model}")
        println(s"   Response: ${completion.message.content.take(100)}")
        completion.usage.foreach { u =>
          println(s"   Tokens: ${u.totalTokens} (${u.promptTokens} prompt + ${u.completionTokens} completion)")
        }

      case Left(error) =>
        println(s" Simple completion FAILED: ${error.formatted}")
    }
  }

  private def testStreaming(client: org.llm4s.llmconnect.LLMClient): Unit = {

    val conversation = Conversation(
      Seq(
        SystemMessage("You are a helpful assistant."),
        UserMessage("Count from 1 to 5, each on new line.")
      )
    )

    var chunkCount  = 0
    val fullContent = new StringBuilder()

    val result = client.streamComplete(
      conversation,
      CompletionOptions(),
      onChunk = { chunk =>
        chunkCount += 1
        chunk.content.foreach { c =>
          fullContent.append(c)
          print(c)
        }
      }
    )

    println()

    result match {
      case Right(completion) =>
        println("\n Streaming SUCCESS")
        println(s"   Chunks received: $chunkCount")
        println(s"   Total content length: ${fullContent.length}")
        println(s"   Content matches completion: ${completion.message.content == fullContent.toString()}")

      case Left(error) =>
        println(s" Streaming FAILED: ${error.formatted}")
    }
  }

  private def testToolCalling(client: org.llm4s.llmconnect.LLMClient): Unit = {

    val toolRegistry = new ToolRegistry(Seq(WeatherTool.tool))

    val conversation = Conversation(
      Seq(
        SystemMessage("Always call the get_weather tool when asked about weather."),
        UserMessage("What's the weather in Paris in celsius?")
      )
    )

    val options = CompletionOptions(tools = Seq(WeatherTool.tool))

    client.complete(conversation, options) match {

      case Right(completion) =>
        if (completion.message.toolCalls.nonEmpty) {

          println(" Tool calling detected")

          completion.message.toolCalls.foreach { tc =>

            val request    = ToolCallRequest(tc.name, tc.arguments)
            val toolResult = toolRegistry.execute(request)

            val updatedConversation =
              conversation
                .addMessage(completion.message)
                .addMessage(ToolMessage(toolResult.map(_.render()).getOrElse("error"), tc.id))

            client.complete(updatedConversation, CompletionOptions()) match {
              case Right(finalCompletion) =>
                println(s"   Final response: ${finalCompletion.message.content.take(200)}")

              case Left(error) =>
                println(s"   Final response failed: ${error.formatted}")
            }
          }

        } else {
          println(" No tool calls in response")
          println(s"   Response: ${completion.message.content.take(200)}")
        }

      case Left(error) =>
        println(s" Tool calling FAILED: ${error.formatted}")
    }
  }
}
