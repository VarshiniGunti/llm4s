package org.llm4s.imagegeneration.provider

import org.llm4s.imagegeneration.{ ImageEditOptions, OpenAIConfig, ValidationError }
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.awt.image.BufferedImage
import java.nio.file.Files
import javax.imageio.ImageIO

class OpenAIImageClientEditTest extends AnyFlatSpec with Matchers {

  "editImage" should "fail when source image does not exist" in {
    val client = new OpenAIImageClient(OpenAIConfig(apiKey = "test-key"))

    val result = client.editImage(
      imagePath = "does-not-exist.png",
      prompt = "add clouds"
    )

    result should matchPattern { case Left(_: ValidationError) => }
  }

  it should "fail when response format is unsupported" in {
    val client = new OpenAIImageClient(OpenAIConfig(apiKey = "test-key"))

    val result = client.editImage(
      imagePath = "does-not-matter.png",
      prompt = "add clouds",
      options = ImageEditOptions(responseFormat = Some("url"))
    )

    result shouldBe Left(ValidationError("Unsupported response format: url"))
  }

  it should "fail when mask dimensions do not match source image dimensions" in {
    val client = new OpenAIImageClient(OpenAIConfig(apiKey = "test-key"))

    val source = Files.createTempFile("openai-source", ".png")
    val mask   = Files.createTempFile("openai-mask", ".png")

    try {
      writePng(source, width = 64, height = 64)
      writePng(mask, width = 32, height = 32)

      val result = client.editImage(
        imagePath = source.toString,
        prompt = "inpaint sky",
        maskPath = Some(mask.toString)
      )

      result should matchPattern { case Left(_: ValidationError) => }
    } finally {
      Files.deleteIfExists(source)
      Files.deleteIfExists(mask)
    }
  }

  private def writePng(path: java.nio.file.Path, width: Int, height: Int): Unit = {
    val image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB)
    ImageIO.write(image, "png", path.toFile)
  }
}
