package org.llm4s.imagegeneration.provider

import org.llm4s.imagegeneration.{ ImageEditOptions, OpenAIConfig, ProviderImageEditOptions, ValidationError }
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.awt.image.BufferedImage
import java.nio.file.Files
import javax.imageio.ImageIO
import scala.util.Using

class OpenAIImageClientEditTest extends AnyFlatSpec with Matchers {

  "editImage" should "fail when source image does not exist" in {
    val client = new OpenAIImageClient(OpenAIConfig(apiKey = "test-key"), HttpClient.create())

    val result = client.editImage(
      imagePath = java.nio.file.Path.of("does-not-exist.png"),
      prompt = "add clouds"
    )

    result should matchPattern { case Left(_: ValidationError) => }
  }

  it should "fail when response format is unsupported" in {
    val client = new OpenAIImageClient(OpenAIConfig(apiKey = "test-key"), HttpClient.create())

    val result = client.editImage(
      imagePath = java.nio.file.Path.of("does-not-matter.png"),
      prompt = "add clouds",
      options = ImageEditOptions(
        providerOptions = Some(
          ProviderImageEditOptions.OpenAI(responseFormat = Some("xml"))
        )
      )
    )

    result shouldBe Left(ValidationError("Unsupported response format for edit: xml"))
  }

  it should "fail when mask dimensions do not match source image dimensions" in {
    val client = new OpenAIImageClient(OpenAIConfig(apiKey = "test-key"), HttpClient.create())

    withTempFiles("openai-source", "openai-mask") { (source, mask) =>
      writePng(source, width = 64, height = 64)
      writePng(mask, width = 32, height = 32)

      val result = client.editImage(
        imagePath = source,
        prompt = "inpaint sky",
        maskPath = Some(mask)
      )

      result should matchPattern { case Left(_: ValidationError) => }
    }
  }

  private def writePng(path: java.nio.file.Path, width: Int, height: Int): Unit = {
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    ImageIO.write(image, "png", path.toFile)
  }

  private def withTempFiles[A](sourcePrefix: String, maskPrefix: String)(
    f: (java.nio.file.Path, java.nio.file.Path) => A
  ): A =
    Using.Manager { use =>
      val source = use(tempFile(sourcePrefix))
      val mask   = use(tempFile(maskPrefix))
      f(source.path, mask.path)
    }.get

  private def tempFile(prefix: String): TempFile =
    TempFile(Files.createTempFile(prefix, ".png"))

  private case class TempFile(path: java.nio.file.Path) extends AutoCloseable {
    override def close(): Unit = Files.deleteIfExists(path)
  }
}
