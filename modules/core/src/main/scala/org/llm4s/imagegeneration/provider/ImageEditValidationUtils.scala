package org.llm4s.imagegeneration.provider

import org.llm4s.imagegeneration.{ ImageGenerationError, ImageSize, ServiceError, ValidationError }

import java.io.File
import java.nio.file.{ Files, Paths }
import javax.imageio.ImageIO
import scala.util.Try

private[provider] object ImageEditValidationUtils {

  def readImageFile(path: String, label: String): Either[ImageGenerationError, Array[Byte]] = {
    val nioPath = Paths.get(path)
    if (!Files.exists(nioPath)) {
      Left(ValidationError(s"$label does not exist at path: $path"))
    } else {
      Try(Files.readAllBytes(nioPath)).toEither.left.map(ex =>
        ServiceError(s"Failed to read $label: ${ex.getMessage}", 500)
      )
    }
  }

  def readImageSize(path: String, label: String): Either[ImageGenerationError, ImageSize] = {
    val file = new File(path)
    if (!file.exists()) {
      Left(ValidationError(s"$label does not exist at path: $path"))
    } else {
      Try(ImageIO.read(file)).toEither.left
        .map(ex => ServiceError(s"Failed to read $label dimensions: ${ex.getMessage}", 500))
        .flatMap {
          case null => Left(ValidationError(s"$label is not a valid image: $path"))
          case img  => Right(toImageSize(img.getWidth, img.getHeight))
        }
    }
  }

  def validateMaskDimensions(
    sourceSize: ImageSize,
    maskPath: Option[String]
  ): Either[ImageGenerationError, Unit] =
    maskPath match {
      case None => Right(())
      case Some(mask) =>
        for {
          maskSize <- readImageSize(mask, "mask image")
          _ <- Either.cond(
            sourceSize == maskSize,
            (),
            ValidationError(
              s"Mask dimensions (${maskSize.width}x${maskSize.height}) must match source image dimensions (${sourceSize.width}x${sourceSize.height})"
            )
          )
        } yield ()
    }

  def toImageSize(width: Int, height: Int): ImageSize =
    (width, height) match {
      case (512, 512)   => ImageSize.Square512
      case (1024, 1024) => ImageSize.Square1024
      case (768, 512)   => ImageSize.Landscape768x512
      case (512, 768)   => ImageSize.Portrait512x768
      case _            => ImageSize.Custom(width, height)
    }
}
