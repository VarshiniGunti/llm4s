package org.llm4s.imagegeneration.provider

import org.llm4s.imagegeneration._
import org.slf4j.LoggerFactory
import upickle.default._

import java.util.Base64
import scala.util.Try

/**
 * Represents the JSON payload for the Stable Diffusion WebUI API's text-to-image endpoint.
 */
case class StableDiffusionPayload(
  prompt: String,
  negative_prompt: String,
  width: Int,
  height: Int,
  steps: Int,
  cfg_scale: Double,
  batch_size: Int,
  n_iter: Int,
  seed: Long,
  sampler_name: String = "Euler a"
)

object StableDiffusionPayload {
  implicit val writer: Writer[StableDiffusionPayload] = macroW
}

case class StableDiffusionImg2ImgPayload(
  prompt: String,
  negative_prompt: String,
  width: Int,
  height: Int,
  steps: Int,
  cfg_scale: Double,
  batch_size: Int,
  n_iter: Int,
  seed: Long,
  sampler_name: String,
  init_images: Seq[String],
  denoising_strength: Double,
  mask: Option[String]
)

object StableDiffusionImg2ImgPayload {
  implicit val writer: Writer[StableDiffusionImg2ImgPayload] = macroW
}

/**
 * Stable Diffusion WebUI API client for image generation and editing.
 *
 * @param config Configuration containing base URL, API key, and timeout settings
 */
class StableDiffusionClient(config: StableDiffusionConfig) extends ImageGenerationClient {

  private val logger = LoggerFactory.getLogger(getClass)

  override def generateImage(
    prompt: String,
    options: ImageGenerationOptions = ImageGenerationOptions()
  ): Either[ImageGenerationError, GeneratedImage] =
    generateImages(prompt, 1, options).flatMap { images =>
      images.headOption.toRight(ValidationError("No images returned from Stable Diffusion txt2img endpoint"))
    }

  override def generateImages(
    prompt: String,
    count: Int,
    options: ImageGenerationOptions = ImageGenerationOptions()
  ): Either[ImageGenerationError, Seq[GeneratedImage]] =
    for {
      _        <- validatePrompt(prompt)
      _        <- validateCount(count)
      payload  <- Right(buildPayload(prompt, count, options))
      response <- Try(makeHttpRequest(payload, "/sdapi/v1/txt2img")).toEither.left.map(UnknownError.apply)
      result   <- parseResponse(response, prompt, options.size, options.format, options.seed)
    } yield result

  override def editImage(
    imagePath: String,
    prompt: String,
    maskPath: Option[String] = None,
    options: ImageEditOptions = ImageEditOptions()
  ): Either[ImageGenerationError, GeneratedImage] =
    for {
      _                 <- validatePrompt(prompt)
      _                 <- validateCount(options.n)
      providerOpts      <- extractStableDiffusionEditOptions(options)
      denoisingStrength <- validateDenoisingStrength(providerOpts.denoisingStrength)
      sourceImage       <- ImageEditValidationUtils.readImageFile(imagePath, "source image")
      sourceSize        <- ImageEditValidationUtils.readImageSize(imagePath, "source image")
      _                 <- ImageEditValidationUtils.validateMaskDimensions(sourceSize, maskPath)
      sourceBase64      <- Right(Base64.getEncoder.encodeToString(sourceImage))
      maskBase64 <- maskPath match {
        case Some(path) =>
          ImageEditValidationUtils
            .readImageFile(path, "mask image")
            .map(bytes => Some(Base64.getEncoder.encodeToString(bytes)))
        case None => Right(None)
      }
      outputSize <- Right(options.size.getOrElse(sourceSize))
      payload <- Right(
        buildImg2ImgPayload(
          prompt = prompt,
          initImage = sourceBase64,
          maskImage = maskBase64,
          options = options,
          outputSize = outputSize,
          denoisingStrength = denoisingStrength
        )
      )
      response <- Try(makeHttpRequest(payload, "/sdapi/v1/img2img")).toEither.left.map(UnknownError.apply)
      images <- parseResponse(
        response = response,
        prompt = prompt,
        size = outputSize,
        format = ImageFormat.PNG,
        seed = None
      )
      image <- images.headOption.toRight(ValidationError("No images returned from Stable Diffusion img2img endpoint"))
    } yield image

  override def health(): Either[ImageGenerationError, ServiceStatus] = Try {
    val response = requests.get(
      s"${config.baseUrl}/sdapi/v1/options",
      readTimeout = 5000,
      connectTimeout = 5000
    )
    response
  }.toEither.left
    .map(e => ServiceError(s"Health check failed: ${e.getMessage}", 0))
    .map { response =>
      if (response.statusCode == 200)
        ServiceStatus(HealthStatus.Healthy, "Stable Diffusion service is responding")
      else ServiceStatus(HealthStatus.Degraded, s"Service returned status code: ${response.statusCode}")
    }

  private def validatePrompt(prompt: String): Either[ImageGenerationError, Unit] =
    Either.cond(prompt.trim.nonEmpty, (), ValidationError("Prompt cannot be empty"))

  private def validateCount(count: Int): Either[ImageGenerationError, Int] =
    Either.cond(count >= 1 && count <= 10, count, ValidationError("Count must be between 1 and 10"))

  private def extractStableDiffusionEditOptions(
    options: ImageEditOptions
  ): Either[ImageGenerationError, ProviderImageEditOptions.StableDiffusion] =
    options.providerOptions match {
      case None => Right(ProviderImageEditOptions.StableDiffusion())
      case Some(stableDiffusion: ProviderImageEditOptions.StableDiffusion) => Right(stableDiffusion)
      case Some(_) =>
        Left(ValidationError("Unsupported provider-specific edit options for Stable Diffusion image client"))
    }

  private def validateDenoisingStrength(value: Option[Double]): Either[ImageGenerationError, Double] = {
    val resolved = value.getOrElse(0.75)
    Either.cond(
      resolved >= 0.0 && resolved <= 1.0,
      resolved,
      ValidationError(s"denoisingStrength must be between 0.0 and 1.0, got: $resolved")
    )
  }

  private def buildPayload(
    prompt: String,
    count: Int,
    options: ImageGenerationOptions
  ): ujson.Value = {
    val payload = StableDiffusionPayload(
      prompt = prompt,
      negative_prompt = options.negativePrompt.getOrElse(""),
      width = options.size.width,
      height = options.size.height,
      steps = options.inferenceSteps,
      cfg_scale = options.guidanceScale,
      batch_size = count,
      n_iter = 1,
      seed = options.seed.getOrElse(-1L),
      sampler_name = options.samplerName.getOrElse("Euler a")
    )
    writeJs(payload)
  }

  private def buildImg2ImgPayload(
    prompt: String,
    initImage: String,
    maskImage: Option[String],
    options: ImageEditOptions,
    outputSize: ImageSize,
    denoisingStrength: Double
  ): ujson.Value = {
    val payload = StableDiffusionImg2ImgPayload(
      prompt = prompt,
      negative_prompt = "",
      width = outputSize.width,
      height = outputSize.height,
      steps = 20,
      cfg_scale = 7.5,
      batch_size = options.n,
      n_iter = 1,
      seed = -1L,
      sampler_name = "Euler a",
      init_images = Seq(initImage),
      denoising_strength = denoisingStrength,
      mask = maskImage
    )
    writeJs(payload)
  }

  private def makeHttpRequest(payload: ujson.Value, endpoint: String): requests.Response = {
    val url = s"${config.baseUrl}$endpoint"
    val headers =
      Map("Content-Type" -> "application/json") ++ config.apiKey.map(key => "Authorization" -> s"Bearer $key")

    logger.debug(s"Making request to: $url")
    logger.debug(s"Payload: ${write(payload, indent = 2)}")

    requests.post(
      url = url,
      data = write(payload),
      headers = headers,
      readTimeout = config.timeout,
      connectTimeout = 10000
    )
  }

  private def parseResponse(
    response: requests.Response,
    prompt: String,
    size: ImageSize,
    format: ImageFormat,
    seed: Option[Long]
  ): Either[ImageGenerationError, Seq[GeneratedImage]] = {

    if (response.statusCode != 200) {
      val errorMsg = s"API request failed with status ${response.statusCode}: ${response.text()}"
      logger.error(errorMsg)
      return Left(ServiceError(errorMsg, response.statusCode))
    }

    Try {
      val responseJson = read[ujson.Value](response.text())
      responseJson("images").arr
    }.toEither.left
      .map(UnknownError.apply)
      .flatMap { images =>
        if (images.isEmpty) Left(ValidationError("No images returned from the API"))
        else {
          val generatedImages = images.map { imageData =>
            GeneratedImage(
              data = imageData.str,
              format = format,
              size = size,
              prompt = prompt,
              seed = seed
            )
          }.toSeq
          logger.info(s"Successfully generated ${generatedImages.length} image(s)")
          Right(generatedImages)
        }
      }
  }

}
