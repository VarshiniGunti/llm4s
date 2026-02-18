package org.llm4s.imagegeneration.provider

import org.llm4s.imagegeneration._
import org.slf4j.LoggerFactory
import ujson._

import java.net.URI
import java.net.http.{ HttpClient => JHttpClient, HttpRequest, HttpResponse }
import java.nio.charset.StandardCharsets
import java.nio.file.{ Files, Paths }
import java.time.{ Duration, Instant }
import java.util.UUID
import scala.util.Try

/**
 * OpenAI Images API client for image generation and editing.
 *
 * @param config Configuration containing API key, model selection, and timeout settings
 */
class OpenAIImageClient(config: OpenAIConfig) extends ImageGenerationClient {

  private val logger            = LoggerFactory.getLogger(getClass)
  private val generationApiUrl  = "https://api.openai.com/v1/images/generations"
  private val imageEditsApiUrl  = "https://api.openai.com/v1/images/edits"
  private val defaultEditFormat = ImageFormat.PNG
  warnIfDeprecatedModelConfigured()

  override def generateImage(
    prompt: String,
    options: ImageGenerationOptions = ImageGenerationOptions()
  ): Either[ImageGenerationError, GeneratedImage] =
    generateImages(prompt, 1, options).flatMap { images =>
      images.headOption.toRight(ValidationError("No images returned from OpenAI image generation endpoint"))
    }

  override def generateImages(
    prompt: String,
    count: Int,
    options: ImageGenerationOptions = ImageGenerationOptions()
  ): Either[ImageGenerationError, Seq[GeneratedImage]] = {
    logger.info(s"Generating $count image(s) with prompt: ${prompt.take(100)}...")
    for {
      validPrompt <- validatePrompt(prompt)
      validCount  <- validateCount(count)
      _           <- validateGenerationOptions(options)
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
      validPrompt   <- validatePrompt(prompt)
      _             <- validateCount(options.n)
      openAIOptions <- extractOpenAIEditOptions(options)
      _             <- validateResponseFormat(openAIOptions.responseFormat, "edit")
      _             <- validateOutputFormat(openAIOptions.outputFormat)
      _             <- validateOutputCompression(openAIOptions.outputCompression)
      sourceImage   <- ImageEditValidationUtils.readImageFile(imagePath, "source image")
      sourceSize    <- ImageEditValidationUtils.readImageSize(imagePath, "source image")
      _             <- ImageEditValidationUtils.validateMaskDimensions(imagePath, maskPath)
      maskImage <- maskPath match {
        case Some(path) => ImageEditValidationUtils.readImageFile(path, "mask image").map(Some(_))
        case None       => Right(None)
      }
      outputSize <- resolveEditOutputSize(options.size, sourceSize)
      _          <- validateEditSize(outputSize)
      responseText <- makeEditApiRequest(
        imagePath = imagePath,
        imageBytes = sourceImage,
        prompt = validPrompt,
        maskPath = maskPath,
        maskBytes = maskImage,
        outputSize = outputSize,
        options = openAIOptions,
        count = options.n
      )
      images <- parseResponse(
        responseText = responseText,
        prompt = validPrompt,
        size = outputSize,
        format = defaultEditFormat,
        seed = None
      )
      image <- images.headOption.toRight(ValidationError("No images returned from OpenAI image edit endpoint"))
    } yield image
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
    else if (prompt.length > maxPromptLength)
      Left(ValidationError(s"Prompt cannot exceed $maxPromptLength characters for ${config.model}"))
    else Right(prompt)

  private def validateCount(count: Int): Either[ImageGenerationError, Int] = {
    val maxCount =
      if (isDallE3Model) 1
      else 10

    if (count < 1 || count > maxCount) {
      Left(ValidationError(s"Count must be between 1 and $maxCount for ${config.model}"))
    } else {
      Right(count)
    }
  }

  private def validateResponseFormat(
    responseFormat: Option[String],
    operation: String
  ): Either[ImageGenerationError, Unit] =
    responseFormat match {
      case None                     => Right(())
      case Some("b64_json" | "url") => Right(())
      case Some(unsupportedResponseType) =>
        Left(ValidationError(s"Unsupported response format for $operation: $unsupportedResponseType"))
    }

  private def validateOutputFormat(outputFormat: Option[String]): Either[ImageGenerationError, Unit] =
    outputFormat match {
      case None                          => Right(())
      case Some("png" | "jpeg" | "webp") => Right(())
      case Some(unsupportedOutputFormat: String) =>
        Left(ValidationError(s"Unsupported output format: $unsupportedOutputFormat"))
    }

  private def validateOutputCompression(outputCompression: Option[Int]): Either[ImageGenerationError, Unit] =
    outputCompression match {
      case None                                      => Right(())
      case Some(level) if level >= 0 && level <= 100 => Right(())
      case Some(level) =>
        Left(ValidationError(s"Output compression must be between 0 and 100, got: $level"))
    }

  private def validateGenerationOptions(options: ImageGenerationOptions): Either[ImageGenerationError, Unit] =
    for {
      _ <- validateResponseFormat(options.responseFormat, "generation")
      _ <- validateOutputFormat(options.outputFormat)
      _ <- validateOutputCompression(options.outputCompression)
      _ <- validateModelOptionCompatibility(options)
    } yield ()

  private def validateModelOptionCompatibility(options: ImageGenerationOptions): Either[ImageGenerationError, Unit] =
    if (
      !isGptImageModel && (options.outputFormat.isDefined || options.outputCompression.isDefined || options.background.isDefined)
    ) {
      Left(
        ValidationError(
          s"outputFormat/outputCompression/background are only supported for GPT Image models; got model ${config.model}"
        )
      )
    } else if (!isDallE3Model && options.style.isDefined) {
      Left(ValidationError(s"style is only supported for dall-e-3; got model ${config.model}"))
    } else {
      Right(())
    }

  private def resolveEditOutputSize(
    requestedSize: Option[ImageSize],
    sourceSize: ImageSize
  ): Either[ImageGenerationError, ImageSize] =
    Right(requestedSize.getOrElse(sourceSize))

  private def validateEditSize(size: ImageSize): Either[ImageGenerationError, Unit] = {
    val allowedSizes = if (isDallE2Model) {
      Set("256x256", "512x512", "1024x1024")
    } else if (isDallE3Model) {
      Set("1024x1024")
    } else {
      Set("1024x1024", "1536x1024", "1024x1536")
    }

    val requested = sizeToApiFormat(size)
    Either.cond(
      allowedSizes.contains(requested),
      (),
      ValidationError(
        s"Unsupported edit size '$requested' for model ${config.model}. Allowed sizes: ${allowedSizes.toSeq.sorted.mkString(", ")}"
      )
    )
  }

  private def extractOpenAIEditOptions(
    options: ImageEditOptions
  ): Either[ImageGenerationError, ProviderImageEditOptions.OpenAI] =
    options.providerOptions match {
      case None                                          => Right(ProviderImageEditOptions.OpenAI())
      case Some(openAI: ProviderImageEditOptions.OpenAI) => Right(openAI)
      case Some(_) =>
        Left(ValidationError("Unsupported provider-specific edit options for OpenAI image client"))
    }

  private def isDallE2Model: Boolean = config.model == "dall-e-2"
  private def isDallE3Model: Boolean = config.model == "dall-e-3"
  private def isGptImageModel: Boolean =
    config.model == "gpt-image-1" || config.model == "gpt-image-1-mini" || config.model.startsWith("gpt-image")
  private def maxPromptLength: Int =
    if (isGptImageModel) 32000
    else if (isDallE2Model) 1000
    else 4000

  private def sizeToApiFormat(size: ImageSize): String =
    size match {
      case ImageSize.Square512 =>
        if (isDallE2Model) "512x512"
        else "1024x1024"
      case ImageSize.Square1024 => "1024x1024"
      case ImageSize.Landscape768x512 =>
        if (isDallE3Model) "1792x1024"
        else if (isDallE2Model) "512x512"
        else "1536x1024"
      case ImageSize.Portrait512x768 =>
        if (isDallE3Model) "1024x1792"
        else if (isDallE2Model) "512x512"
        else "1024x1536"
      case ImageSize.Custom(0, 0) => "auto"
      case ImageSize.Custom(w, h) => s"${w}x${h}"
    }

  private def makeGenerationApiRequest(
    prompt: String,
    count: Int,
    options: ImageGenerationOptions
  ): Either[ImageGenerationError, requests.Response] = {
    val responseFormat: String = options.responseFormat.getOrElse("b64_json")
    val requestBody = Obj(
      "model"           -> Str(config.model),
      "prompt"          -> Str(prompt),
      "n"               -> Num(count.toDouble),
      "size"            -> Str(sizeToApiFormat(options.size)),
      "response_format" -> Str(responseFormat)
    )

    options.quality.foreach(v => requestBody("quality") = Str(v))
    options.style.foreach(v => requestBody("style") = Str(v))
    options.background.foreach(v => requestBody("background") = Str(v))
    options.outputFormat.foreach(v => requestBody("output_format") = Str(v))
    options.outputCompression.foreach(v => requestBody("output_compression") = Num(v.toDouble))
    options.user.foreach(v => requestBody("user") = Str(v))

    if (isDallE3Model && !requestBody.obj.contains("quality")) {
      requestBody("quality") = "standard"
    }
    if (isGptImageModel && !requestBody.obj.contains("quality")) {
      requestBody("quality") = "medium"
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
    options: ProviderImageEditOptions.OpenAI,
    count: Int
  ): Either[ImageGenerationError, String] = {
    val boundary       = s"llm4s-${UUID.randomUUID().toString}"
    val responseFormat = options.responseFormat.getOrElse("b64_json")
    val size           = sizeToApiFormat(outputSize)

    val fields = Map(
      "model"           -> config.model,
      "prompt"          -> prompt,
      "n"               -> count.toString,
      "size"            -> size,
      "response_format" -> responseFormat
    ) ++ options.quality.map(value => "quality" -> value) ++
      options.style.map(value => "style" -> value) ++
      options.background.map(value => "background" -> value) ++
      options.outputFormat.map(value => "output_format" -> value) ++
      options.outputCompression.map(value => "output_compression" -> value.toString) ++
      options.user.map(value => "user" -> value)

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
      val maybeB64 = imageData.obj.get("b64_json").collect { case Str(value) => value }
      val maybeUrl = imageData.obj.get("url").collect { case Str(value) => value }
      GeneratedImage(
        data = maybeB64.getOrElse(""),
        format = format,
        size = size,
        createdAt = Instant.now(),
        prompt = prompt,
        seed = seed,
        filePath = None,
        url = maybeUrl
      )
    }.toSeq

    if (images.isEmpty) {
      Left(ValidationError("No images returned from OpenAI API response"))
    } else {
      logger.info(s"Successfully generated ${images.length} image(s)")
      Right(images)
    }
  }

  private def warnIfDeprecatedModelConfigured(): Unit =
    if (isDallE2Model || isDallE3Model) {
      logger.warn(
        s"${config.model} is deprecated and scheduled for removal on 2026-05-12. Migrate to gpt-image-1, gpt-image-1-mini, or gpt-image-1.5."
      )
    }
}
