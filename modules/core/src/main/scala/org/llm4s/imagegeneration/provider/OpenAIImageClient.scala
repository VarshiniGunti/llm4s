package org.llm4s.imagegeneration.provider

import org.llm4s.imagegeneration._
import org.slf4j.LoggerFactory
import ujson._

import java.io.File
import java.net.URI
import java.net.http.{ HttpClient => JHttpClient, HttpRequest, HttpResponse }
import java.nio.charset.StandardCharsets
import java.nio.file.{ Files, Paths }
import java.time.{ Duration, Instant }
import java.util.UUID
import javax.imageio.ImageIO
import scala.util.Try

/**
 * OpenAI DALL-E API client for image generation and editing.
 *
 * @param config Configuration containing API key, model selection, and timeout settings
 */
class OpenAIImageClient(config: OpenAIConfig) extends ImageGenerationClient {

  private val logger            = LoggerFactory.getLogger(getClass)
  private val generationApiUrl  = "https://api.openai.com/v1/images/generations"
  private val imageEditsApiUrl  = "https://api.openai.com/v1/images/edits"
  private val defaultEditFormat = ImageFormat.PNG

  override def generateImage(
    prompt: String,
    options: ImageGenerationOptions = ImageGenerationOptions()
  ): Either[ImageGenerationError, GeneratedImage] =
    generateImages(prompt, 1, options).map(_.head)

  override def generateImages(
    prompt: String,
    count: Int,
    options: ImageGenerationOptions = ImageGenerationOptions()
  ): Either[ImageGenerationError, Seq[GeneratedImage]] = {
    logger.info(s"Generating $count image(s) with prompt: ${prompt.take(100)}...")
    for {
      validPrompt <- validatePrompt(prompt)
      validCount  <- validateCount(count)
      response    <- makeGenerationApiRequest(validPrompt, validCount, options)
      images      <- parseResponse(response.text(), validPrompt, options.size, options.format, options.seed)
    } yield images
  }

  override def editImage(
    imagePath: String,
    prompt: String,
    maskPath: Option[String] = None,
    options: ImageEditOptions = ImageEditOptions()
  ): Either[ImageGenerationError, GeneratedImage] = {
    logger.info(s"Editing image with prompt: ${prompt.take(100)}...")
    for {
      validPrompt <- validatePrompt(prompt)
      _           <- validateCount(options.n)
      _           <- validateResponseFormat(options.responseFormat)
      sourceImage <- readImageFile(imagePath, "source image")
      sourceSize  <- readImageSize(imagePath, "source image")
      _           <- validateMaskDimensions(imagePath, maskPath)
      maskImage <- maskPath match {
        case Some(path) => readImageFile(path, "mask image").map(Some(_))
        case None       => Right(None)
      }
      outputSize <- resolveEditOutputSize(options.size, sourceSize)
      responseText <- makeEditApiRequest(
        imagePath = imagePath,
        imageBytes = sourceImage,
        prompt = validPrompt,
        maskPath = maskPath,
        maskBytes = maskImage,
        outputSize = outputSize,
        options = options
      )
      images <- parseResponse(
        responseText = responseText,
        prompt = validPrompt,
        size = outputSize,
        format = defaultEditFormat,
        seed = None
      )
    } yield images.head
  }

  override def health(): Either[ImageGenerationError, ServiceStatus] = {
    val response = requests.get(
      "https://api.openai.com/v1/models",
      headers = Map("Authorization" -> s"Bearer ${config.apiKey}"),
      readTimeout = 5000,
      connectTimeout = 5000
    )

    if (response.statusCode == 200) {
      Right(ServiceStatus(status = HealthStatus.Healthy, message = "OpenAI API is responding"))
    } else if (response.statusCode == 429) {
      Right(ServiceStatus(status = HealthStatus.Degraded, message = "Rate limited but operational"))
    } else {
      Right(ServiceStatus(status = HealthStatus.Unhealthy, message = s"API returned status ${response.statusCode}"))
    }
  }

  private def validatePrompt(prompt: String): Either[ImageGenerationError, String] =
    if (prompt.trim.isEmpty) Left(ValidationError("Prompt cannot be empty"))
    else if (prompt.length > 4000) Left(ValidationError("Prompt cannot exceed 4000 characters"))
    else Right(prompt)

  private def validateCount(count: Int): Either[ImageGenerationError, Int] = {
    val maxCount = if (config.model == "dall-e-3") 1 else 10
    if (count < 1 || count > maxCount) {
      Left(ValidationError(s"Count must be between 1 and $maxCount for ${config.model}"))
    } else {
      Right(count)
    }
  }

  private def validateResponseFormat(responseFormat: Option[String]): Either[ImageGenerationError, Unit] =
    responseFormat match {
      case None                  => Right(())
      case Some("b64_json")      => Right(())
      case Some(unsupportedType) => Left(ValidationError(s"Unsupported response format: $unsupportedType"))
    }

  private def validateMaskDimensions(imagePath: String, maskPath: Option[String]): Either[ImageGenerationError, Unit] =
    maskPath match {
      case None => Right(())
      case Some(mask) =>
        for {
          sourceSize <- readImageSize(imagePath, "source image")
          maskSize   <- readImageSize(mask, "mask image")
          _ <- Either.cond(
            sourceSize == maskSize,
            (),
            ValidationError(
              s"Mask dimensions (${maskSize.width}x${maskSize.height}) must match source image dimensions (${sourceSize.width}x${sourceSize.height})"
            )
          )
        } yield ()
    }

  private def readImageFile(path: String, label: String): Either[ImageGenerationError, Array[Byte]] = {
    val nioPath = Paths.get(path)
    if (!Files.exists(nioPath)) {
      Left(ValidationError(s"$label does not exist at path: $path"))
    } else {
      Try(Files.readAllBytes(nioPath)).toEither.left.map(ex =>
        ServiceError(s"Failed to read $label: ${ex.getMessage}", 500)
      )
    }
  }

  private def readImageSize(path: String, label: String): Either[ImageGenerationError, ImageSize] = {
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

  private def resolveEditOutputSize(
    requestedSize: Option[ImageSize],
    sourceSize: ImageSize
  ): Either[ImageGenerationError, ImageSize] =
    Right(requestedSize.getOrElse(sourceSize))

  private def toImageSize(width: Int, height: Int): ImageSize =
    (width, height) match {
      case (512, 512)   => ImageSize.Square512
      case (1024, 1024) => ImageSize.Square1024
      case (768, 512)   => ImageSize.Landscape768x512
      case (512, 768)   => ImageSize.Portrait512x768
      case _            => ImageSize.Custom(width, height)
    }

  private def sizeToApiFormat(size: ImageSize): String =
    size match {
      case ImageSize.Square512        => if (config.model == "dall-e-3") "1024x1024" else "512x512"
      case ImageSize.Square1024       => "1024x1024"
      case ImageSize.Landscape768x512 => if (config.model == "dall-e-3") "1792x1024" else "512x512"
      case ImageSize.Portrait512x768  => if (config.model == "dall-e-3") "1024x1792" else "512x512"
      case ImageSize.Custom(w, h)     => s"${w}x${h}"
    }

  private def makeGenerationApiRequest(
    prompt: String,
    count: Int,
    options: ImageGenerationOptions
  ): Either[ImageGenerationError, requests.Response] = {
    val requestBody = Obj(
      "model"           -> config.model,
      "prompt"          -> prompt,
      "n"               -> count,
      "size"            -> sizeToApiFormat(options.size),
      "response_format" -> "b64_json"
    )

    if (config.model == "dall-e-3") {
      requestBody("quality") = "standard"
    }

    val response = requests.post(
      generationApiUrl,
      headers = Map(
        "Authorization" -> s"Bearer ${config.apiKey}",
        "Content-Type"  -> "application/json"
      ),
      data = requestBody.toString,
      readTimeout = config.timeout,
      connectTimeout = 10000
    )

    if (response.statusCode == 200) Right(response) else handleErrorResponse(response.statusCode, response.text())
  }

  private def makeEditApiRequest(
    imagePath: String,
    imageBytes: Array[Byte],
    prompt: String,
    maskPath: Option[String],
    maskBytes: Option[Array[Byte]],
    outputSize: ImageSize,
    options: ImageEditOptions
  ): Either[ImageGenerationError, String] = {
    val boundary       = s"llm4s-${UUID.randomUUID().toString}"
    val responseFormat = options.responseFormat.getOrElse("b64_json")
    val size           = options.size.map(sizeToApiFormat).getOrElse(sizeToApiFormat(outputSize))

    val fields = Map(
      "model"           -> config.model,
      "prompt"          -> prompt,
      "n"               -> options.n.toString,
      "size"            -> size,
      "response_format" -> responseFormat
    ) ++ options.quality.map(value => "quality" -> value)

    val body = buildMultipartBody(
      boundary = boundary,
      fields = fields,
      imagePath = imagePath,
      imageBytes = imageBytes,
      maskPath = maskPath,
      maskBytes = maskBytes
    )

    val request = HttpRequest
      .newBuilder(URI.create(imageEditsApiUrl))
      .timeout(Duration.ofMillis(config.timeout.toLong))
      .header("Authorization", s"Bearer ${config.apiKey}")
      .header("Content-Type", s"multipart/form-data; boundary=$boundary")
      .POST(HttpRequest.BodyPublishers.ofByteArray(body))
      .build()

    val httpClient = JHttpClient.newBuilder().connectTimeout(Duration.ofSeconds(10)).build()
    val response = Try(httpClient.send(request, HttpResponse.BodyHandlers.ofString())).toEither.left
      .map(ex => ServiceError(s"OpenAI image edit request failed: ${ex.getMessage}", 500))

    response.flatMap { resp =>
      if (resp.statusCode() == 200) Right(resp.body()) else handleErrorResponse(resp.statusCode(), resp.body())
    }
  }

  private def buildMultipartBody(
    boundary: String,
    fields: Map[String, String],
    imagePath: String,
    imageBytes: Array[Byte],
    maskPath: Option[String],
    maskBytes: Option[Array[Byte]]
  ): Array[Byte] = {
    val newline = "\r\n"
    val out     = new java.io.ByteArrayOutputStream()

    fields.foreach { case (name, value) =>
      out.write(s"--$boundary$newline".getBytes(StandardCharsets.UTF_8))
      out.write(s"""Content-Disposition: form-data; name="$name"$newline$newline""".getBytes(StandardCharsets.UTF_8))
      out.write(value.getBytes(StandardCharsets.UTF_8))
      out.write(newline.getBytes(StandardCharsets.UTF_8))
    }

    writeFilePart(out, boundary, "image", imagePath, imageBytes, newline)

    (maskPath, maskBytes) match {
      case (Some(path), Some(bytes)) => writeFilePart(out, boundary, "mask", path, bytes, newline)
      case _                         =>
    }

    out.write(s"--$boundary--$newline".getBytes(StandardCharsets.UTF_8))
    out.toByteArray
  }

  private def writeFilePart(
    out: java.io.ByteArrayOutputStream,
    boundary: String,
    fieldName: String,
    filePath: String,
    fileBytes: Array[Byte],
    newline: String
  ): Unit = {
    val filename = Paths.get(filePath).getFileName.toString
    val mimeType = Option(Files.probeContentType(Paths.get(filePath))).getOrElse("application/octet-stream")

    out.write(s"--$boundary$newline".getBytes(StandardCharsets.UTF_8))
    out.write(
      s"""Content-Disposition: form-data; name="$fieldName"; filename="$filename"$newline""".getBytes(
        StandardCharsets.UTF_8
      )
    )
    out.write(s"Content-Type: $mimeType$newline$newline".getBytes(StandardCharsets.UTF_8))
    out.write(fileBytes)
    out.write(newline.getBytes(StandardCharsets.UTF_8))
  }

  private def handleErrorResponse(
    statusCode: Int,
    responseBody: String
  ): Either[ImageGenerationError, Nothing] = {
    val errorMessage = Try {
      val json = read(responseBody)
      json("error")("message").str
    }.getOrElse(responseBody)

    statusCode match {
      case 401  => Left(AuthenticationError("Invalid API key"))
      case 429  => Left(RateLimitError("Rate limit exceeded"))
      case 400  => Left(ValidationError(s"Invalid request: $errorMessage"))
      case code => Left(ServiceError(s"API error: $errorMessage", code))
    }
  }

  private def parseResponse(
    responseText: String,
    prompt: String,
    size: ImageSize,
    format: ImageFormat,
    seed: Option[Long]
  ): Either[ImageGenerationError, Seq[GeneratedImage]] = {
    val json       = read(responseText)
    val imagesData = json("data").arr

    val images = imagesData.map { imageData =>
      GeneratedImage(
        data = imageData("b64_json").str,
        format = format,
        size = size,
        createdAt = Instant.now(),
        prompt = prompt,
        seed = seed,
        filePath = None
      )
    }.toSeq

    logger.info(s"Successfully generated ${images.length} image(s)")
    Right(images)
  }
}
