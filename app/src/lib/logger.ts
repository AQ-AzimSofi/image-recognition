const LOG_MSG_PREFIXES = {
  DETECTION: "[Detection]",
  REKOGNITION: "[Rekognition]",
  S3: "[S3]",
} as const;

type LogLevel = "debug" | "info" | "warn" | "error";

class Logger {
  private prefix: string;

  constructor(prefix: string = "") {
    this.prefix = prefix;
  }

  private formatMessage(message: string): string {
    return this.prefix ? `${this.prefix} ${message}` : message;
  }

  private log(level: LogLevel, message: string, obj?: Record<string, any>) {
    const formatted = this.formatMessage(message);
    if (obj) {
      console[level](formatted, JSON.stringify(obj, null, 2));
    } else {
      console[level](formatted);
    }
  }

  debug(message: string, obj?: Record<string, any>) {
    this.log("debug", message, obj);
  }

  info(message: string, obj?: Record<string, any>) {
    this.log("info", message, obj);
  }

  warn(message: string, obj?: Record<string, any>) {
    this.log("warn", message, obj);
  }

  error(message: string, obj?: Record<string, any>) {
    this.log("error", message, obj);
  }
}

export const logger = new Logger();
export const detectionLogger = new Logger(LOG_MSG_PREFIXES.DETECTION);
export const rekognitionLogger = new Logger(LOG_MSG_PREFIXES.REKOGNITION);
export const s3Logger = new Logger(LOG_MSG_PREFIXES.S3);
