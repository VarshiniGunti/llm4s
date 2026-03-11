package org.llm4s.knowledgegraph.graphrag

import org.llm4s.knowledgegraph.storage.InMemoryGraphStore
import org.llm4s.knowledgegraph.{ Edge, Node }
import org.llm4s.llmconnect.LLMClient
import org.llm4s.llmconnect.model._
import org.llm4s.types.Result
import org.llm4s.vectorstore.HybridSearchResult
import org.scalatest.funsuite.AnyFunSuite
import org.scalatest.matchers.should.Matchers

class GraphRAGSpec extends AnyFunSuite with Matchers {

  private class DeterministicLLM extends LLMClient {
    override def complete(conversation: Conversation, options: CompletionOptions): Result[Completion] = {
      val system = conversation.messages.collectFirst { case s: SystemMessage => s.content }.getOrElse("")
      val content =
        if (system.contains("Summarize the graph community")) "finance themes and relationships"
        else if (system.contains("Answer the question using only the provided community summary.")) "partial answer"
        else if (system.contains("Synthesize a single coherent answer from partial answers.")) "global answer"
        else if (system.contains("local graph neighborhood")) "local answer"
        else if (system.contains("hybrid vector and graph evidence")) "hybrid answer"
        else "generic answer"

      Right(
        Completion(
          id = "test",
          created = 1L,
          content = content,
          model = "test-model",
          message = AssistantMessage(contentOpt = Some(content)),
          usage = None
        )
      )
    }

    override def streamComplete(
      conversation: Conversation,
      options: CompletionOptions,
      onChunk: StreamedChunk => Unit
    ): Result[Completion] =
      complete(conversation, options)

    override def getContextWindow(): Int = 8192
    override def getReserveCompletion(): Int = 1024
  }

  private def seededStore(): InMemoryGraphStore = {
    val store = new InMemoryGraphStore()
    val nodes = Seq(
      Node("alice", "Person", Map("name" -> ujson.Str("Alice"), "source" -> ujson.Str("doc-a"))),
      Node("bob", "Person", Map("name" -> ujson.Str("Bob"), "source" -> ujson.Str("doc-a"))),
      Node("fintech", "Org", Map("name" -> ujson.Str("FinTech Labs"), "source" -> ujson.Str("doc-a"))),
      Node("carol", "Person", Map("name" -> ujson.Str("Carol"), "source" -> ujson.Str("doc-b"))),
      Node("dave", "Person", Map("name" -> ujson.Str("Dave"), "source" -> ujson.Str("doc-b"))),
      Node("bank", "Org", Map("name" -> ujson.Str("Bank Group"), "source" -> ujson.Str("doc-b")))
    )
    nodes.foreach(node => store.upsertNode(node).isRight shouldBe true)

    val edges = Seq(
      Edge("alice", "bob", "KNOWS"),
      Edge("alice", "fintech", "WORKS_AT"),
      Edge("bob", "fintech", "WORKS_AT"),
      Edge("carol", "dave", "KNOWS"),
      Edge("carol", "bank", "WORKS_AT"),
      Edge("dave", "bank", "WORKS_AT")
    )
    edges.foreach(edge => store.upsertEdge(edge).isRight shouldBe true)
    store
  }

  test("detectCommunities builds multi-level community hierarchy") {
    val rag = new GraphRAG(seededStore(), new DeterministicLLM())

    val hierarchy = rag.detectCommunities().toOption.get
    hierarchy.levels.nonEmpty shouldBe true
    hierarchy.levels.head.size should be >= 2
    hierarchy.levels.head.flatMap(_.nodeIds).toSet should contain allElementsOf Set(
      "alice",
      "bob",
      "fintech",
      "carol",
      "dave",
      "bank"
    )
  }

  test("summarizeCommunities generates summaries and persists community nodes") {
    val store = seededStore()
    val rag   = new GraphRAG(store, new DeterministicLLM())

    val hierarchy = rag.summarizeCommunities().toOption.get
    hierarchy.allCommunities.forall(_.summary.exists(_.nonEmpty)) shouldBe true

    val graph = store.loadAll().toOption.get
    graph.nodes.values.exists(_.label == "Community") shouldBe true
    graph.edges.exists(_.relationship == "CONTAINS") shouldBe true
  }

  test("globalSearch uses community summaries map-reduce") {
    val rag = new GraphRAG(seededStore(), new DeterministicLLM())

    val result = rag.globalSearch("What are the finance themes?", topK = 2).toOption.get
    result.answer shouldBe "global answer"
    result.rankedCommunities.nonEmpty shouldBe true
    result.partialAnswers.nonEmpty shouldBe true
  }

  test("localSearch fans out from seed entities") {
    val rag = new GraphRAG(seededStore(), new DeterministicLLM())

    val result = rag.localSearch("What does Alice do?", maxDepth = 1).toOption.get
    result.answer shouldBe "local answer"
    result.seedNodeIds should contain("alice")
    result.visitedNodeIds should contain("bob")
    result.visitedNodeIds should contain("fintech")
  }

  test("hybridSearch combines vector signal with graph traversal") {
    val rag = new GraphRAG(seededStore(), new DeterministicLLM())
    val vectorResults = Seq(
      HybridSearchResult(
        id = "doc-a-chunk-0",
        content = "Alice works at FinTech Labs",
        score = 0.95,
        metadata = Map("entityId" -> "alice", "docId" -> "doc-a")
      )
    )

    val result = rag.hybridSearch("Who is connected to Alice?", vectorResults).toOption.get
    result.answer shouldBe "hybrid answer"
    result.seedNodeIds should contain("alice")
    result.hits.nonEmpty shouldBe true
    result.hits.head.node.id shouldBe "alice"
  }
}
