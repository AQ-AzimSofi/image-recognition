export const formatErrorForLogging = (error: unknown) => {
  let message = error instanceof Error ? error.message : String(error);
  if (
    !message ||
    message.trim() === "" ||
    message === "undefined" ||
    message === "null"
  ) {
    message = "Unknown error";
  }
  return {
    message,
    stack: error instanceof Error ? error.stack : undefined,
  };
};
