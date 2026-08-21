import api from "./axios";
import type { SessionAnalyticsData } from "../types/analytics";

export async function getAnalytics(
  conversationId: string
): Promise<SessionAnalyticsData> {
  try {
    const response = await api.get(
      `/conversations/${conversationId}/analytics`
    );
    return response.data;
  } catch (error: any) {
    if (error.response?.data?.detail) {
      throw new Error(error.response.data.detail);
    }
    throw new Error("Unable to fetch session analytics.");
  }
}
