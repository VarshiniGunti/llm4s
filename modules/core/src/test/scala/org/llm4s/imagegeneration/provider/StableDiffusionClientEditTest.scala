package org.llm4s.imagegeneration.provider

import org.llm4s.imagegeneration.{ ImageEditOptions, StableDiffusionConfig, ValidationError }
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.awt.image.BufferedImage
import java.nio.file.Files
import javax.imageio.ImageIO

class StableDiffusionClientEditTest extends AnyFlatSpec with Matchers {

  "editImage" should "fail when source image does not exist" in {
    val client = new StableDiffusionClient(StableDiffusionConfig(baseUrl = "http://localhost:7860"))

    val result = client.editImage(
      imagePath = "does-not-exist.png",
      prompt = "add clouds"
    )

    result should matchPattern { case Left(_: ValidationError) => }
  }

  it should "fail when mask dimensions do not match source image dimensions" in {
    val client = new StableDiffusionClient(StableDiffusionConfig(baseUrl = "http://localhost:7860"))

    val source = Files.createTempFile("sd-source", ".png")
    val mask   = Files.createTempFile("sd-mask", ".png")

    try {
      writePng(source, width = 128, height = 128)
      writePng(mask, width = 64, height = 64)

      val result = client.editImage(
        imagePath = source.toString,
        prompt = "remove foreground object",
        maskPath = Some(mask.toString),
        options = ImageEditOptions(n = 1)
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
